"""
Rule Loader for Game Loop Rules Engine.
Handles loading rules from various sources (YAML, JSON, database).
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import uuid4, UUID

from .rule_models import (
    Rule,
    RuleAction,
    RuleCondition,
    ActionType,
    ConditionOperator,
    RulePriority,
)


class RuleLoader:
    """Loads rules from various sources and formats."""

    def __init__(self):
        """Initialize the rule loader."""
        pass

    def load_from_file(self, file_path: Union[str, Path]) -> List[Rule]:
        """
        Load rules from a file (YAML or JSON).

        Args:
            file_path: Path to the rules file

        Returns:
            List of loaded rules

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Rules file not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")

            if file_path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(content)
            elif file_path.suffix.lower() == ".json":
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            return self._parse_rules_data(data)

        except Exception as e:
            raise ValueError(f"Error loading rules from {file_path}: {str(e)}")

    def load_from_dict(self, rules_data: Dict) -> List[Rule]:
        """
        Load rules from a dictionary.

        Args:
            rules_data: Dictionary containing rules data

        Returns:
            List of loaded rules
        """
        return self._parse_rules_data(rules_data)

    def load_from_string(
        self, rules_string: str, format_type: str = "yaml"
    ) -> List[Rule]:
        """
        Load rules from a string.

        Args:
            rules_string: String containing rules data
            format_type: Format of the string ('yaml' or 'json')

        Returns:
            List of loaded rules
        """
        try:
            if format_type.lower() in ["yml", "yaml"]:
                data = yaml.safe_load(rules_string)
            elif format_type.lower() == "json":
                data = json.loads(rules_string)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            return self._parse_rules_data(data)

        except Exception as e:
            raise ValueError(f"Error loading rules from string: {str(e)}")

    def _parse_rules_data(self, data: Dict) -> List[Rule]:
        """
        Parse rules data from a dictionary structure.

        Args:
            data: Dictionary containing rules data

        Returns:
            List of parsed rules
        """
        rules = []

        # Handle both single rule and multiple rules formats
        if "rules" in data:
            rules_list = data["rules"]
        elif "rule" in data:
            rules_list = [data["rule"]]
        elif isinstance(data, list):
            rules_list = data
        else:
            # Assume data is a single rule
            rules_list = [data]

        for rule_data in rules_list:
            try:
                rule = self._parse_single_rule(rule_data)
                rules.append(rule)
            except Exception as e:
                # Log error but continue with other rules
                print(f"Error parsing rule: {str(e)}")
                continue

        return rules

    def _parse_single_rule(self, rule_data: Dict) -> Rule:
        """
        Parse a single rule from dictionary data.

        Args:
            rule_data: Dictionary containing single rule data

        Returns:
            Parsed Rule object
        """
        # Parse basic rule properties
        rule_id_raw = rule_data.get("id", str(uuid4()))
        
        # Convert string ID to UUID if needed
        if isinstance(rule_id_raw, str):
            try:
                rule_id = UUID(rule_id_raw)
            except ValueError:
                # Invalid UUID string, generate new one
                rule_id = uuid4()
        else:
            rule_id = rule_id_raw
            
        name = rule_data.get("name", f"rule_{str(rule_id)[:8]}")
        description = rule_data.get("description")
        enabled = rule_data.get("enabled", True)
        tags = rule_data.get("tags", [])

        # Parse priority
        priority_str = rule_data.get("priority", "medium").upper()
        try:
            priority = RulePriority[priority_str]
        except KeyError:
            priority = RulePriority.MEDIUM

        # Parse conditions
        conditions = []
        conditions_data = rule_data.get("conditions", [])
        if not isinstance(conditions_data, list):
            conditions_data = [conditions_data]

        for condition_data in conditions_data:
            condition = self._parse_condition(condition_data)
            conditions.append(condition)

        # Parse actions
        actions = []
        actions_data = rule_data.get("actions", [])
        if not isinstance(actions_data, list):
            actions_data = [actions_data]

        for action_data in actions_data:
            action = self._parse_action(action_data)
            actions.append(action)

        return Rule(
            id=rule_id,
            name=name,
            description=description,
            priority=priority,
            enabled=enabled,
            conditions=conditions,
            actions=actions,
            tags=tags,
            created_at=rule_data.get("created_at"),
            modified_at=rule_data.get("modified_at"),
        )

    def _parse_condition(self, condition_data: Dict) -> RuleCondition:
        """
        Parse a rule condition from dictionary data.

        Args:
            condition_data: Dictionary containing condition data

        Returns:
            Parsed RuleCondition object
        """
        field_path = condition_data.get("field", condition_data.get("field_path", ""))
        operator_str = condition_data.get("operator", "equals").lower()
        expected_value = condition_data.get(
            "value", condition_data.get("expected_value")
        )
        description = condition_data.get("description")

        # Map operator string to enum
        operator_mapping = {
            "equals": ConditionOperator.EQUALS,
            "eq": ConditionOperator.EQUALS,
            "==": ConditionOperator.EQUALS,
            "not_equals": ConditionOperator.NOT_EQUALS,
            "ne": ConditionOperator.NOT_EQUALS,
            "!=": ConditionOperator.NOT_EQUALS,
            "greater_than": ConditionOperator.GREATER_THAN,
            "gt": ConditionOperator.GREATER_THAN,
            ">": ConditionOperator.GREATER_THAN,
            "less_than": ConditionOperator.LESS_THAN,
            "lt": ConditionOperator.LESS_THAN,
            "<": ConditionOperator.LESS_THAN,
            "greater_equal": ConditionOperator.GREATER_EQUAL,
            "ge": ConditionOperator.GREATER_EQUAL,
            ">=": ConditionOperator.GREATER_EQUAL,
            "less_equal": ConditionOperator.LESS_EQUAL,
            "le": ConditionOperator.LESS_EQUAL,
            "<=": ConditionOperator.LESS_EQUAL,
            "contains": ConditionOperator.CONTAINS,
            "not_contains": ConditionOperator.NOT_CONTAINS,
            "in": ConditionOperator.IN,
            "not_in": ConditionOperator.NOT_IN,
            "exists": ConditionOperator.EXISTS,
            "not_exists": ConditionOperator.NOT_EXISTS,
            "regex_match": ConditionOperator.REGEX_MATCH,
            "regex": ConditionOperator.REGEX_MATCH,
        }

        operator = operator_mapping.get(operator_str, ConditionOperator.EQUALS)

        return RuleCondition(
            field_path=field_path,
            operator=operator,
            expected_value=expected_value,
            description=description,
        )

    def _parse_action(self, action_data: Dict) -> RuleAction:
        """
        Parse a rule action from dictionary data.

        Args:
            action_data: Dictionary containing action data

        Returns:
            Parsed RuleAction object
        """
        action_type_str = action_data.get(
            "type", action_data.get("action_type", "custom")
        ).lower()
        target_path = action_data.get("target", action_data.get("target_path"))
        parameters = action_data.get("parameters", action_data.get("params", {}))
        description = action_data.get("description")

        # Map action type string to enum
        action_type_mapping = {
            "modify_state": ActionType.MODIFY_STATE,
            "modify": ActionType.MODIFY_STATE,
            "set": ActionType.MODIFY_STATE,
            "send_message": ActionType.SEND_MESSAGE,
            "message": ActionType.SEND_MESSAGE,
            "send": ActionType.SEND_MESSAGE,
            "trigger_event": ActionType.TRIGGER_EVENT,
            "trigger": ActionType.TRIGGER_EVENT,
            "event": ActionType.TRIGGER_EVENT,
            "block_action": ActionType.BLOCK_ACTION,
            "block": ActionType.BLOCK_ACTION,
            "prevent": ActionType.BLOCK_ACTION,
            "grant_reward": ActionType.GRANT_REWARD,
            "reward": ActionType.GRANT_REWARD,
            "grant": ActionType.GRANT_REWARD,
            "impose_penalty": ActionType.IMPOSE_PENALTY,
            "penalty": ActionType.IMPOSE_PENALTY,
            "punish": ActionType.IMPOSE_PENALTY,
            "change_location": ActionType.CHANGE_LOCATION,
            "move": ActionType.CHANGE_LOCATION,
            "teleport": ActionType.CHANGE_LOCATION,
            "spawn_entity": ActionType.SPAWN_ENTITY,
            "spawn": ActionType.SPAWN_ENTITY,
            "create": ActionType.SPAWN_ENTITY,
            "despawn_entity": ActionType.DESPAWN_ENTITY,
            "despawn": ActionType.DESPAWN_ENTITY,
            "remove": ActionType.DESPAWN_ENTITY,
            "custom": ActionType.CUSTOM,
        }

        action_type = action_type_mapping.get(action_type_str, ActionType.CUSTOM)

        return RuleAction(
            action_type=action_type,
            target_path=target_path,
            parameters=parameters,
            description=description,
        )

    def create_sample_rules_file(self, file_path: Union[str, Path]) -> None:
        """
        Create a sample rules file for reference.

        Args:
            file_path: Path where the sample file should be created
        """
        sample_rules = {
            "rules": [
                {
                    "name": "low_health_warning",
                    "description": "Warn player when health is low",
                    "priority": "high",
                    "conditions": [
                        {
                            "field": "player_state.health",
                            "operator": "less_than",
                            "value": 20,
                            "description": "Health below 20",
                        }
                    ],
                    "actions": [
                        {
                            "type": "send_message",
                            "parameters": {
                                "message": "Warning: Your health is critically low!",
                                "style": "warning",
                            },
                            "description": "Send warning message",
                        }
                    ],
                    "tags": ["health", "warning"],
                },
                {
                    "name": "inventory_full_block",
                    "description": "Prevent taking items when inventory is full",
                    "priority": "medium",
                    "conditions": [
                        {
                            "field": "current_action",
                            "operator": "equals",
                            "value": "take",
                        },
                        {
                            "field": "player_state.inventory_count",
                            "operator": "greater_equal",
                            "value": 10,
                        },
                    ],
                    "actions": [
                        {
                            "type": "block_action",
                            "parameters": {"reason": "Your inventory is full!"},
                        }
                    ],
                    "tags": ["inventory", "blocking"],
                },
            ]
        }

        file_path = Path(file_path)

        if file_path.suffix.lower() in [".yml", ".yaml"]:
            content = yaml.dump(sample_rules, default_flow_style=False, indent=2)
        else:
            content = json.dumps(sample_rules, indent=2)

        file_path.write_text(content, encoding="utf-8")
