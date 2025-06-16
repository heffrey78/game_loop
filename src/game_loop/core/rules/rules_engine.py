"""
Rules Engine for Game Loop.
Main orchestrator for rule evaluation, conflict resolution, and execution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from .rule_evaluator import RuleEvaluator
from .rule_loader import RuleLoader
from .rule_models import (
    Rule,
    RuleConflict,
    RuleEngineConfig,
    RuleEvaluationContext,
    RulePriority,
    RuleResult,
)


class RulesEngine:
    """Main rules engine that orchestrates rule evaluation and execution."""

    def __init__(self, config: Optional[RuleEngineConfig] = None):
        """
        Initialize the rules engine.

        Args:
            config: Configuration for the rules engine
        """
        self.config = config or RuleEngineConfig()
        self.evaluator = RuleEvaluator()
        self.loader = RuleLoader()

        # Rule storage
        self._rules: Dict[UUID, Rule] = {}
        self._rules_by_name: Dict[str, Rule] = {}
        self._rules_by_priority: Dict[RulePriority, List[Rule]] = {
            priority: [] for priority in RulePriority
        }
        self._rules_by_tag: Dict[str, List[Rule]] = {}

        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0

    def load_rules_from_file(self, file_path: str) -> int:
        """
        Load rules from a file.

        Args:
            file_path: Path to the rules file

        Returns:
            Number of rules loaded
        """
        rules = self.loader.load_from_file(file_path)
        return self.add_rules(rules)

    def load_rules_from_dict(self, rules_data: Dict) -> int:
        """
        Load rules from a dictionary.

        Args:
            rules_data: Dictionary containing rules data

        Returns:
            Number of rules loaded
        """
        rules = self.loader.load_from_dict(rules_data)
        return self.add_rules(rules)

    def add_rule(self, rule: Rule) -> bool:
        """
        Add a single rule to the engine.

        Args:
            rule: The rule to add

        Returns:
            True if rule was added successfully, False otherwise
        """
        try:
            # Check for name conflicts
            if rule.name in self._rules_by_name:
                existing_rule = self._rules_by_name[rule.name]
                if existing_rule.id != rule.id:
                    raise ValueError(f"Rule with name '{rule.name}' already exists")

            # Add to storage
            self._rules[rule.id] = rule
            self._rules_by_name[rule.name] = rule
            self._rules_by_priority[rule.priority].append(rule)

            # Add to tag index
            for tag in rule.tags:
                if tag not in self._rules_by_tag:
                    self._rules_by_tag[tag] = []
                self._rules_by_tag[tag].append(rule)

            return True

        except Exception:
            return False

    def add_rules(self, rules: List[Rule]) -> int:
        """
        Add multiple rules to the engine.

        Args:
            rules: List of rules to add

        Returns:
            Number of rules successfully added
        """
        added_count = 0
        for rule in rules:
            if self.add_rule(rule):
                added_count += 1
        return added_count

    def remove_rule(self, rule_id: UUID) -> bool:
        """
        Remove a rule from the engine.

        Args:
            rule_id: ID of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]

        # Remove from all indexes
        del self._rules[rule_id]
        del self._rules_by_name[rule.name]
        self._rules_by_priority[rule.priority].remove(rule)

        for tag in rule.tags:
            if tag in self._rules_by_tag:
                self._rules_by_tag[tag].remove(rule)
                if not self._rules_by_tag[tag]:
                    del self._rules_by_tag[tag]

        return True

    def get_rule(self, rule_id: UUID) -> Optional[Rule]:
        """
        Get a rule by ID.

        Args:
            rule_id: ID of the rule

        Returns:
            The rule if found, None otherwise
        """
        return self._rules.get(rule_id)

    def get_rule_by_name(self, name: str) -> Optional[Rule]:
        """
        Get a rule by name.

        Args:
            name: Name of the rule

        Returns:
            The rule if found, None otherwise
        """
        return self._rules_by_name.get(name)

    def get_rules_by_tag(self, tag: str) -> List[Rule]:
        """
        Get all rules with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of rules with the tag
        """
        return self._rules_by_tag.get(tag, []).copy()

    def get_rules_by_priority(self, priority: RulePriority) -> List[Rule]:
        """
        Get all rules with a specific priority.

        Args:
            priority: Priority level to filter by

        Returns:
            List of rules with the priority
        """
        return self._rules_by_priority[priority].copy()

    def evaluate_rules(
        self, context: RuleEvaluationContext, tags: Optional[List[str]] = None
    ) -> List[RuleResult]:
        """
        Evaluate all applicable rules against the provided context.

        Args:
            context: The evaluation context
            tags: Optional list of tags to filter rules (if None, all rules are evaluated)

        Returns:
            List of rule evaluation results
        """
        import time

        start_time = time.time()

        # Get rules to evaluate
        rules_to_evaluate = self._get_applicable_rules(tags)

        # Sort by priority
        rules_to_evaluate.sort(key=lambda r: r.priority.value)

        # Limit number of rules if configured
        if len(rules_to_evaluate) > self.config.max_rules_per_evaluation:
            rules_to_evaluate = rules_to_evaluate[
                : self.config.max_rules_per_evaluation
            ]

        results = []

        # Evaluate each rule
        for rule in rules_to_evaluate:
            try:
                result = self.evaluator.evaluate_rule(rule, context)
                results.append(result)

                # Check for timeout
                if self.config.rule_timeout_ms > 0:
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms > self.config.rule_timeout_ms:
                        break

            except Exception as e:
                # Create error result
                error_result = RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    triggered=False,
                    conditions_met=False,
                    error_message=str(e),
                )
                results.append(error_result)

        # Update performance tracking
        self._evaluation_count += 1
        evaluation_time = (time.time() - start_time) * 1000
        self._total_evaluation_time += evaluation_time

        # Detect conflicts if enabled
        if self.config.enable_conflict_detection:
            conflicts = self._detect_conflicts(results)
            if conflicts:
                results = self._resolve_conflicts(results, conflicts)

        return results

    def _get_applicable_rules(self, tags: Optional[List[str]] = None) -> List[Rule]:
        """
        Get rules that are applicable for evaluation.

        Args:
            tags: Optional list of tags to filter rules

        Returns:
            List of applicable rules
        """
        if tags is None:
            # Return all rules (enabled and disabled will be handled during evaluation)
            return list(self._rules.values())

        # Get rules matching any of the provided tags
        applicable_rules = []
        seen_ids = set()
        for tag in tags:
            for rule in self.get_rules_by_tag(tag):
                if rule.id not in seen_ids:
                    seen_ids.add(rule.id)
                    applicable_rules.append(rule)

        # Return all applicable rules (enabled and disabled will be handled during evaluation)
        return applicable_rules

    def _detect_conflicts(self, results: List[RuleResult]) -> List[RuleConflict]:
        """
        Detect conflicts between triggered rules.

        Args:
            results: List of rule evaluation results

        Returns:
            List of detected conflicts
        """
        conflicts = []
        triggered_results = [r for r in results if r.triggered]

        # Simple conflict detection - rules that modify the same target
        # This is a placeholder implementation
        for i, result1 in enumerate(triggered_results):
            for result2 in triggered_results[i + 1 :]:
                rule1 = self._rules.get(result1.rule_id)
                rule2 = self._rules.get(result2.rule_id)

                if rule1 and rule2:
                    # Check if rules have conflicting actions
                    if self._rules_conflict(rule1, rule2):
                        conflict = RuleConflict(
                            conflicting_rules=[rule1.id, rule2.id],
                            conflict_type="action_conflict",
                            description=f"Rules '{rule1.name}' and '{rule2.name}' have conflicting actions",
                        )
                        conflicts.append(conflict)

        return conflicts

    def _rules_conflict(self, rule1: Rule, rule2: Rule) -> bool:
        """
        Check if two rules have conflicting actions.

        Args:
            rule1: First rule
            rule2: Second rule

        Returns:
            True if rules conflict, False otherwise
        """
        # Placeholder implementation - check if both rules modify state
        rule1_modifies = any(
            action.action_type.value == "modify_state" for action in rule1.actions
        )
        rule2_modifies = any(
            action.action_type.value == "modify_state" for action in rule2.actions
        )

        return rule1_modifies and rule2_modifies

    def _resolve_conflicts(
        self, results: List[RuleResult], conflicts: List[RuleConflict]
    ) -> List[RuleResult]:
        """
        Resolve conflicts between rules using priority-based resolution.

        Args:
            results: Original rule evaluation results
            conflicts: List of detected conflicts

        Returns:
            Results with conflicts resolved
        """
        resolved_results = []
        conflicted_rule_ids = set()

        # Collect all conflicted rule IDs
        for conflict in conflicts:
            conflicted_rule_ids.update(conflict.conflicting_rules)

        # For each conflict, keep only the highest priority rule
        for conflict in conflicts:
            highest_priority_rule = None
            highest_priority_value = float("inf")

            for rule_id in conflict.conflicting_rules:
                rule = self._rules.get(rule_id)
                if rule:
                    priority_val = rule.priority.value
                    if priority_val < highest_priority_value:
                        highest_priority_rule = rule_id
                        highest_priority_value = priority_val

            # Mark the resolution
            conflict.resolution_strategy = "priority_based"
            conflict.resolved_rule_id = highest_priority_rule

        # Build resolved results list
        for result in results:
            if result.rule_id in conflicted_rule_ids:
                # Only include if this is the winning rule in a conflict
                is_winner = any(
                    conflict.resolved_rule_id == result.rule_id
                    for conflict in conflicts
                    if result.rule_id in conflict.conflicting_rules
                )
                if is_winner:
                    resolved_results.append(result)
            else:
                resolved_results.append(result)

        return resolved_results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the rules engine.

        Returns:
            Dictionary containing engine statistics
        """
        total_rules = len(self._rules)
        enabled_rules = sum(1 for rule in self._rules.values() if rule.enabled)

        priority_counts = {
            priority.name: len(rules)
            for priority, rules in self._rules_by_priority.items()
        }

        avg_evaluation_time = (
            self._total_evaluation_time / self._evaluation_count
            if self._evaluation_count > 0
            else 0
        )

        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "priority_distribution": priority_counts,
            "tags": list(self._rules_by_tag.keys()),
            "evaluation_count": self._evaluation_count,
            "avg_evaluation_time_ms": avg_evaluation_time,
            "total_evaluation_time_ms": self._total_evaluation_time,
        }

    def clear_rules(self) -> None:
        """Clear all rules from the engine."""
        self._rules.clear()
        self._rules_by_name.clear()
        for priority_list in self._rules_by_priority.values():
            priority_list.clear()
        self._rules_by_tag.clear()

    def export_rules(self, file_path: str, format_type: str = "yaml") -> bool:
        """
        Export all rules to a file.

        Args:
            file_path: Path where rules should be exported
            format_type: Format for export ('yaml' or 'json')

        Returns:
            True if export successful, False otherwise
        """
        try:
            rules_data = {"rules": [rule.model_dump() for rule in self._rules.values()]}

            file_path_obj = Path(file_path)

            if format_type.lower() in ["yml", "yaml"]:
                import yaml

                content = yaml.dump(rules_data, default_flow_style=False, indent=2)
            else:
                import json

                content = json.dumps(rules_data, indent=2, default=str)

            file_path_obj.write_text(content, encoding="utf-8")
            return True

        except Exception:
            return False
