"""
Game Rules Manager for Game Loop.
Integrates the rules engine with the game loop and provides game-specific functionality.
"""

from pathlib import Path
from typing import Any

from .rule_models import (
    RuleEngineConfig,
    RuleEvaluationContext,
    RuleResult,
)
from .rule_triggers import RuleTriggerManager
from .rules_engine import RulesEngine


class GameRulesManager:
    """Manages rules integration with the game loop."""

    def __init__(
        self,
        config: RuleEngineConfig | None = None,
        rules_directory: str | None = None,
    ):
        """
        Initialize the game rules manager.

        Args:
            config: Configuration for the rules engine
            rules_directory: Directory containing rule files to load
        """
        self.config = config or RuleEngineConfig()
        self.rules_engine = RulesEngine(self.config)
        self.trigger_manager = RuleTriggerManager(self.rules_engine)

        # Game state integration
        self._game_state_manager = None
        self._output_generator = None

        # Load default rules if directory provided
        if rules_directory:
            self.load_rules_directory(rules_directory)

    def set_game_dependencies(self, game_state_manager, output_generator):
        """
        Set dependencies on game components.

        Args:
            game_state_manager: Game state manager instance
            output_generator: Output generator instance
        """
        self._game_state_manager = game_state_manager
        self._output_generator = output_generator

    def load_rules_directory(self, directory_path: str) -> int:
        """
        Load all rule files from a directory.

        Args:
            directory_path: Path to directory containing rule files

        Returns:
            Total number of rules loaded
        """
        rules_dir = Path(directory_path)
        if not rules_dir.exists():
            return 0

        total_loaded = 0

        # Load YAML and JSON files
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            for rule_file in rules_dir.glob(pattern):
                try:
                    loaded = self.rules_engine.load_rules_from_file(str(rule_file))
                    total_loaded += loaded
                except Exception as e:
                    # Log error but continue loading other files
                    print(f"Error loading rules from {rule_file}: {e}")

        return total_loaded

    def evaluate_action_rules(
        self,
        action: str,
        action_parameters: dict[str, Any],
        player_state: dict[str, Any],
        world_state: dict[str, Any],
        location_data: dict[str, Any],
    ) -> list[RuleResult]:
        """
        Evaluate rules for a player action.

        Args:
            action: The action being performed
            action_parameters: Parameters of the action
            player_state: Current player state
            world_state: Current world state
            location_data: Current location data

        Returns:
            List of rule evaluation results
        """
        context = RuleEvaluationContext(
            player_state=player_state,
            world_state=world_state,
            current_action=action,
            action_parameters=action_parameters,
            location_data=location_data,
            custom_data={},
        )

        # Evaluate rules with action-related tags
        return self.rules_engine.evaluate_rules(
            context, tags=["action", "blocking", action]
        )

    def evaluate_state_change_rules(
        self,
        player_state: dict[str, Any],
        world_state: dict[str, Any],
        changes: dict[str, Any],
    ) -> list[RuleResult]:
        """
        Evaluate rules for state changes.

        Args:
            player_state: Current player state
            world_state: Current world state
            changes: Dictionary of what changed

        Returns:
            List of rule evaluation results
        """
        context = RuleEvaluationContext(
            player_state=player_state,
            world_state=world_state,
            custom_data={"changes": changes},
        )

        # Determine appropriate tags based on what changed
        tags = ["state_change"]
        if "health" in changes:
            tags.append("health")
        if "inventory" in changes:
            tags.append("inventory")
        if "level" in changes:
            tags.append("level")

        return self.rules_engine.evaluate_rules(context, tags=tags)

    def process_game_event(
        self, event_type: str, event_data: dict[str, Any]
    ) -> list[RuleResult]:
        """
        Process a game event through the trigger system.

        Args:
            event_type: Type of event that occurred
            event_data: Data about the event

        Returns:
            List of rule evaluation results
        """
        return self.trigger_manager.process_event(event_type, event_data)

    def check_action_allowed(
        self,
        action: str,
        action_parameters: dict[str, Any],
        player_state: dict[str, Any],
        world_state: dict[str, Any],
        location_data: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Check if an action is allowed based on rules.

        Args:
            action: The action being attempted
            action_parameters: Parameters of the action
            player_state: Current player state
            world_state: Current world state
            location_data: Current location data

        Returns:
            Tuple of (is_allowed, block_reason)
        """
        results = self.evaluate_action_rules(
            action, action_parameters, player_state, world_state, location_data
        )

        # Check if any rule blocked the action
        for result in results:
            if result.triggered:
                rule = self.rules_engine.get_rule(result.rule_id)
                if rule:
                    for action_obj in rule.actions:
                        if action_obj.action_type.value == "block_action":
                            reason = action_obj.parameters.get(
                                "reason", "Action blocked by rule"
                            )
                            return False, reason

        return True, None

    def apply_rule_results(self, results: list[RuleResult]) -> None:
        """
        Apply the effects of triggered rules.

        Args:
            results: List of rule evaluation results to apply
        """
        for result in results:
            if result.triggered and result.error_message is None:
                self._apply_single_rule_result(result)

    def _apply_single_rule_result(self, result: RuleResult) -> None:
        """
        Apply the effects of a single triggered rule.

        Args:
            result: Rule evaluation result to apply
        """
        rule = self.rules_engine.get_rule(result.rule_id)
        if not rule:
            return

        for action in rule.actions:
            try:
                self._execute_rule_action(action, result)
            except Exception as e:
                # Log error but continue with other actions
                print(f"Error executing rule action: {e}")

    def _execute_rule_action(self, action, result: RuleResult) -> None:
        """
        Execute a specific rule action.

        Args:
            action: The rule action to execute
            result: The rule result context
        """
        action_type = action.action_type.value

        if action_type == "send_message" and self._output_generator:
            message = action.parameters.get("message", "")
            style = action.parameters.get("style", "info")

            if style == "warning":
                self._output_generator.display_error(message, "warning")
            elif style == "success":
                self._output_generator.display_system_message(message, "success")
            else:
                self._output_generator.display_system_message(message, "info")

        elif action_type == "modify_state" and self._game_state_manager:
            # This would require implementing state modification in the game state manager
            # For now, this is a placeholder
            target_path = action.target_path
            parameters = action.parameters
            # self._game_state_manager.modify_state(target_path, parameters)

        elif action_type == "trigger_event":
            event_type = action.parameters.get("event_type")
            event_data = action.parameters.get("data", {})

            if event_type:
                # Trigger the event through the trigger manager
                self.process_game_event(event_type, event_data)

        # Other action types would be implemented based on game needs

    def add_custom_rule_from_dict(self, rule_data: dict[str, Any]) -> bool:
        """
        Add a custom rule from dictionary data.

        Args:
            rule_data: Dictionary containing rule definition

        Returns:
            True if rule was added successfully, False otherwise
        """
        try:
            rules = self.rules_engine.loader.load_from_dict(rule_data)
            return self.rules_engine.add_rules(rules) > 0
        except Exception:
            return False

    def get_rules_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """
        Get rules with a specific tag as dictionaries.

        Args:
            tag: Tag to filter by

        Returns:
            List of rule dictionaries
        """
        rules = self.rules_engine.get_rules_by_tag(tag)
        return [rule.model_dump() for rule in rules]

    def disable_rule(self, rule_name: str) -> bool:
        """
        Disable a rule by name.

        Args:
            rule_name: Name of the rule to disable

        Returns:
            True if rule was disabled, False if not found
        """
        rule = self.rules_engine.get_rule_by_name(rule_name)
        if rule:
            rule.enabled = False
            return True
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """
        Enable a rule by name.

        Args:
            rule_name: Name of the rule to enable

        Returns:
            True if rule was enabled, False if not found
        """
        rule = self.rules_engine.get_rule_by_name(rule_name)
        if rule:
            rule.enabled = True
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the rules system.

        Returns:
            Dictionary containing rules system statistics
        """
        engine_stats = self.rules_engine.get_statistics()
        trigger_stats = self.trigger_manager.get_statistics()

        return {
            "rules_engine": engine_stats,
            "trigger_manager": trigger_stats,
            "integration": {
                "has_game_state_manager": self._game_state_manager is not None,
                "has_output_generator": self._output_generator is not None,
            },
        }

    def export_all_rules(self, file_path: str, format_type: str = "yaml") -> bool:
        """
        Export all rules to a file.

        Args:
            file_path: Path where rules should be exported
            format_type: Format for export ('yaml' or 'json')

        Returns:
            True if export successful, False otherwise
        """
        return self.rules_engine.export_rules(file_path, format_type)
