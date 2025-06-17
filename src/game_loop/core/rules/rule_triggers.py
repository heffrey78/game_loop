"""
Rule Triggers for Game Loop Rules Engine.
Handles automatic rule evaluation based on game events and state changes.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from .rule_models import RuleEvaluationContext
from .rules_engine import RulesEngine


class TriggerType(Enum):
    """Types of triggers that can activate rule evaluation."""

    ACTION_PERFORMED = "action_performed"
    STATE_CHANGED = "state_changed"
    LOCATION_CHANGED = "location_changed"
    INVENTORY_CHANGED = "inventory_changed"
    TIME_ELAPSED = "time_elapsed"
    HEALTH_CHANGED = "health_changed"
    CONVERSATION_STARTED = "conversation_started"
    QUEST_UPDATED = "quest_updated"
    CUSTOM_EVENT = "custom_event"


class RuleTrigger:
    """Represents a trigger that can activate rule evaluation."""

    def __init__(
        self,
        trigger_type: TriggerType,
        rule_tags: list[str] | None = None,
        condition: Callable[[dict[str, Any]], bool] | None = None,
        description: str | None = None,
    ):
        """
        Initialize a rule trigger.

        Args:
            trigger_type: Type of trigger
            rule_tags: Tags of rules to evaluate when triggered (None = all rules)
            condition: Optional condition function to check before triggering
            description: Human-readable description
        """
        self.id = uuid4()
        self.trigger_type = trigger_type
        self.rule_tags = rule_tags or []
        self.condition = condition
        self.description = description
        self.enabled = True
        self.trigger_count = 0

    def should_trigger(self, event_data: dict[str, Any]) -> bool:
        """
        Check if this trigger should activate for the given event.

        Args:
            event_data: Data about the event

        Returns:
            True if trigger should activate, False otherwise
        """
        if not self.enabled:
            return False

        if self.condition:
            try:
                return self.condition(event_data)
            except Exception:
                return False

        return True

    def __str__(self) -> str:
        return f"Trigger({self.trigger_type.value}, tags={self.rule_tags})"


class RuleTriggerManager:
    """Manages rule triggers and automatic rule evaluation."""

    def __init__(self, rules_engine: RulesEngine):
        """
        Initialize the trigger manager.

        Args:
            rules_engine: The rules engine to use for evaluation
        """
        self.rules_engine = rules_engine
        self._triggers: dict[UUID, RuleTrigger] = {}
        self._triggers_by_type: dict[TriggerType, list[RuleTrigger]] = {
            trigger_type: [] for trigger_type in TriggerType
        }

        # Event listeners
        self._event_listeners: dict[str, list[Callable]] = {}

        # Setup default triggers
        self._setup_default_triggers()

    def add_trigger(self, trigger: RuleTrigger) -> bool:
        """
        Add a trigger to the manager.

        Args:
            trigger: The trigger to add

        Returns:
            True if trigger was added successfully, False otherwise
        """
        try:
            self._triggers[trigger.id] = trigger
            self._triggers_by_type[trigger.trigger_type].append(trigger)
            return True
        except Exception:
            return False

    def remove_trigger(self, trigger_id: UUID) -> bool:
        """
        Remove a trigger from the manager.

        Args:
            trigger_id: ID of the trigger to remove

        Returns:
            True if trigger was removed, False if not found
        """
        if trigger_id not in self._triggers:
            return False

        trigger = self._triggers[trigger_id]
        del self._triggers[trigger_id]
        self._triggers_by_type[trigger.trigger_type].remove(trigger)
        return True

    def get_trigger(self, trigger_id: UUID) -> RuleTrigger | None:
        """
        Get a trigger by ID.

        Args:
            trigger_id: ID of the trigger

        Returns:
            The trigger if found, None otherwise
        """
        return self._triggers.get(trigger_id)

    def get_triggers_by_type(self, trigger_type: TriggerType) -> list[RuleTrigger]:
        """
        Get all triggers of a specific type.

        Args:
            trigger_type: Type of trigger

        Returns:
            List of triggers of the specified type
        """
        return self._triggers_by_type[trigger_type].copy()

    def process_event(self, event_type: str, event_data: dict[str, Any]) -> list[Any]:
        """
        Process a game event and trigger appropriate rule evaluations.

        Args:
            event_type: Type of event that occurred
            event_data: Data about the event

        Returns:
            List of rule evaluation results
        """
        results = []

        # Map event type to trigger type
        trigger_type = self._map_event_to_trigger_type(event_type)
        if not trigger_type:
            return results

        # Get applicable triggers
        triggers = self.get_triggers_by_type(trigger_type)

        # Process each trigger
        for trigger in triggers:
            if trigger.should_trigger(event_data):
                trigger.trigger_count += 1

                # Create evaluation context
                context = self._create_evaluation_context(event_data)

                # Evaluate rules
                rule_results = self.rules_engine.evaluate_rules(
                    context, tags=trigger.rule_tags if trigger.rule_tags else None
                )

                results.extend(rule_results)

        return results

    def _map_event_to_trigger_type(self, event_type: str) -> TriggerType | None:
        """
        Map an event type string to a TriggerType enum.

        Args:
            event_type: String event type

        Returns:
            Corresponding TriggerType or None if no mapping
        """
        event_mapping = {
            "action_performed": TriggerType.ACTION_PERFORMED,
            "action_complete": TriggerType.ACTION_PERFORMED,
            "state_changed": TriggerType.STATE_CHANGED,
            "player_state_changed": TriggerType.STATE_CHANGED,
            "location_changed": TriggerType.LOCATION_CHANGED,
            "move": TriggerType.LOCATION_CHANGED,
            "inventory_changed": TriggerType.INVENTORY_CHANGED,
            "item_taken": TriggerType.INVENTORY_CHANGED,
            "item_dropped": TriggerType.INVENTORY_CHANGED,
            "time_elapsed": TriggerType.TIME_ELAPSED,
            "health_changed": TriggerType.HEALTH_CHANGED,
            "conversation_started": TriggerType.CONVERSATION_STARTED,
            "talk": TriggerType.CONVERSATION_STARTED,
            "quest_updated": TriggerType.QUEST_UPDATED,
            "quest_completed": TriggerType.QUEST_UPDATED,
        }

        return event_mapping.get(event_type.lower())

    def _create_evaluation_context(
        self, event_data: dict[str, Any]
    ) -> RuleEvaluationContext:
        """
        Create a rule evaluation context from event data.

        Args:
            event_data: Data about the event

        Returns:
            RuleEvaluationContext for rule evaluation
        """
        # Extract common context data from event
        return RuleEvaluationContext(
            player_state=event_data.get("player_state"),
            world_state=event_data.get("world_state"),
            session_data=event_data.get("session_data"),
            current_action=event_data.get("action"),
            action_parameters=event_data.get("action_parameters", {}),
            current_location=event_data.get("location_id"),
            location_data=event_data.get("location_data"),
            game_time=event_data.get("game_time"),
            real_time=event_data.get("timestamp"),
            custom_data=event_data.get("custom_data", {}),
        )

    def _setup_default_triggers(self) -> None:
        """Setup default triggers for common game events."""

        # Action performed trigger
        action_trigger = RuleTrigger(
            trigger_type=TriggerType.ACTION_PERFORMED,
            description="Triggered when any player action is performed",
        )
        self.add_trigger(action_trigger)

        # Health changed trigger (for low health warnings)
        health_trigger = RuleTrigger(
            trigger_type=TriggerType.HEALTH_CHANGED,
            rule_tags=["health", "warning"],
            description="Triggered when player health changes",
        )
        self.add_trigger(health_trigger)

        # Inventory changed trigger
        inventory_trigger = RuleTrigger(
            trigger_type=TriggerType.INVENTORY_CHANGED,
            rule_tags=["inventory"],
            description="Triggered when player inventory changes",
        )
        self.add_trigger(inventory_trigger)

        # Location changed trigger
        location_trigger = RuleTrigger(
            trigger_type=TriggerType.LOCATION_CHANGED,
            rule_tags=["location", "movement"],
            description="Triggered when player changes location",
        )
        self.add_trigger(location_trigger)

        # Quest updated trigger
        quest_trigger = RuleTrigger(
            trigger_type=TriggerType.QUEST_UPDATED,
            rule_tags=["quest"],
            description="Triggered when quest status changes",
        )
        self.add_trigger(quest_trigger)

    def add_event_listener(self, event_type: str, callback: Callable) -> None:
        """
        Add an event listener for custom processing.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def remove_event_listener(self, event_type: str, callback: Callable) -> bool:
        """
        Remove an event listener.

        Args:
            event_type: Type of event
            callback: Function to remove

        Returns:
            True if listener was removed, False if not found
        """
        if event_type in self._event_listeners:
            try:
                self._event_listeners[event_type].remove(callback)
                return True
            except ValueError:
                pass
        return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the trigger manager.

        Returns:
            Dictionary containing trigger statistics
        """
        total_triggers = len(self._triggers)
        enabled_triggers = sum(
            1 for trigger in self._triggers.values() if trigger.enabled
        )

        trigger_type_counts = {
            trigger_type.value: len(triggers)
            for trigger_type, triggers in self._triggers_by_type.items()
            if triggers
        }

        total_trigger_count = sum(
            trigger.trigger_count for trigger in self._triggers.values()
        )

        return {
            "total_triggers": total_triggers,
            "enabled_triggers": enabled_triggers,
            "disabled_triggers": total_triggers - enabled_triggers,
            "trigger_type_distribution": trigger_type_counts,
            "total_activations": total_trigger_count,
            "event_listeners": {
                event_type: len(listeners)
                for event_type, listeners in self._event_listeners.items()
            },
        }
