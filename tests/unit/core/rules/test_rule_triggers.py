"""Tests for rule triggers."""

from unittest.mock import Mock

import pytest

from game_loop.core.rules.rule_models import (
    ActionType,
    ConditionOperator,
    Rule,
    RuleAction,
    RuleCondition,
)
from game_loop.core.rules.rule_triggers import (
    RuleTrigger,
    RuleTriggerManager,
    TriggerType,
)
from game_loop.core.rules.rules_engine import RulesEngine


class TestRuleTrigger:
    """Test RuleTrigger class."""

    def test_create_basic_trigger(self):
        """Test creating a basic trigger."""
        trigger = RuleTrigger(
            trigger_type=TriggerType.ACTION_PERFORMED,
            rule_tags=["action"],
            description="Action trigger",
        )

        assert trigger.trigger_type == TriggerType.ACTION_PERFORMED
        assert trigger.rule_tags == ["action"]
        assert trigger.description == "Action trigger"
        assert trigger.enabled is True
        assert trigger.trigger_count == 0
        assert trigger.id is not None

    def test_trigger_without_condition(self):
        """Test trigger without condition always triggers."""
        trigger = RuleTrigger(trigger_type=TriggerType.HEALTH_CHANGED)

        # Should trigger for any event data
        assert trigger.should_trigger({"health": 50}) is True
        assert trigger.should_trigger({}) is True

    def test_trigger_with_condition(self):
        """Test trigger with custom condition."""

        def health_low_condition(event_data):
            return event_data.get("health", 100) < 30

        trigger = RuleTrigger(
            trigger_type=TriggerType.HEALTH_CHANGED, condition=health_low_condition
        )

        # Should trigger only when health is low
        assert trigger.should_trigger({"health": 20}) is True
        assert trigger.should_trigger({"health": 50}) is False
        assert trigger.should_trigger({}) is False  # No health data

    def test_disabled_trigger(self):
        """Test that disabled trigger doesn't trigger."""
        trigger = RuleTrigger(trigger_type=TriggerType.ACTION_PERFORMED)
        trigger.enabled = False

        assert trigger.should_trigger({"action": "test"}) is False

    def test_trigger_with_error_in_condition(self):
        """Test trigger handles condition errors gracefully."""

        def error_condition(event_data):
            raise ValueError("Test error")

        trigger = RuleTrigger(
            trigger_type=TriggerType.ACTION_PERFORMED, condition=error_condition
        )

        # Should not trigger if condition raises error
        assert trigger.should_trigger({"action": "test"}) is False

    def test_trigger_string_representation(self):
        """Test trigger string representation."""
        trigger = RuleTrigger(
            trigger_type=TriggerType.INVENTORY_CHANGED, rule_tags=["inventory", "items"]
        )

        trigger_str = str(trigger)
        assert "inventory_changed" in trigger_str
        assert "inventory" in trigger_str
        assert "items" in trigger_str


class TestRuleTriggerManager:
    """Test RuleTriggerManager class."""

    @pytest.fixture
    def rules_engine(self):
        """Create a mock rules engine for testing."""
        return Mock(spec=RulesEngine)

    @pytest.fixture
    def trigger_manager(self, rules_engine):
        """Create a trigger manager for testing."""
        return RuleTriggerManager(rules_engine)

    def test_manager_initialization(self, trigger_manager):
        """Test trigger manager initialization."""
        assert trigger_manager.rules_engine is not None
        assert len(trigger_manager._triggers) == 5  # Default triggers are created

        # Should have default triggers for each type
        for trigger_type in TriggerType:
            assert trigger_type in trigger_manager._triggers_by_type

    def test_manager_has_default_triggers(self, trigger_manager):
        """Test that manager creates default triggers."""
        # Should have some default triggers
        total_triggers = sum(
            len(triggers) for triggers in trigger_manager._triggers_by_type.values()
        )
        assert total_triggers > 0

        # Should have action trigger
        action_triggers = trigger_manager.get_triggers_by_type(
            TriggerType.ACTION_PERFORMED
        )
        assert len(action_triggers) >= 1

    def test_add_custom_trigger(self, trigger_manager):
        """Test adding a custom trigger."""
        custom_trigger = RuleTrigger(
            trigger_type=TriggerType.CUSTOM_EVENT,
            rule_tags=["custom"],
            description="Custom test trigger",
        )

        result = trigger_manager.add_trigger(custom_trigger)

        assert result is True
        assert custom_trigger.id in trigger_manager._triggers
        assert (
            custom_trigger
            in trigger_manager._triggers_by_type[TriggerType.CUSTOM_EVENT]
        )

    def test_remove_trigger(self, trigger_manager):
        """Test removing a trigger."""
        # Add a trigger first
        custom_trigger = RuleTrigger(trigger_type=TriggerType.CUSTOM_EVENT)
        trigger_manager.add_trigger(custom_trigger)

        # Remove it
        result = trigger_manager.remove_trigger(custom_trigger.id)

        assert result is True
        assert custom_trigger.id not in trigger_manager._triggers
        assert (
            custom_trigger
            not in trigger_manager._triggers_by_type[TriggerType.CUSTOM_EVENT]
        )

    def test_remove_nonexistent_trigger(self, trigger_manager):
        """Test removing nonexistent trigger returns False."""
        from uuid import uuid4

        result = trigger_manager.remove_trigger(uuid4())
        assert result is False

    def test_get_trigger_by_id(self, trigger_manager):
        """Test getting trigger by ID."""
        custom_trigger = RuleTrigger(trigger_type=TriggerType.CUSTOM_EVENT)
        trigger_manager.add_trigger(custom_trigger)

        retrieved_trigger = trigger_manager.get_trigger(custom_trigger.id)
        assert retrieved_trigger == custom_trigger

        from uuid import uuid4

        nonexistent_trigger = trigger_manager.get_trigger(uuid4())
        assert nonexistent_trigger is None


class TestEventProcessing:
    """Test event processing and rule triggering."""

    @pytest.fixture
    def rules_engine(self):
        """Create a rules engine with sample rules."""
        engine = RulesEngine()

        # Add a health warning rule
        health_rule = Rule(
            name="health_warning",
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=30,
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

        # Add an inventory rule
        inventory_rule = Rule(
            name="inventory_full",
            conditions=[
                RuleCondition(
                    field_path="current_action",
                    operator=ConditionOperator.EQUALS,
                    expected_value="take",
                ),
                RuleCondition(
                    field_path="player_state.inventory_count",
                    operator=ConditionOperator.GREATER_EQUAL,
                    expected_value=10,
                ),
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.BLOCK_ACTION,
                    parameters={"reason": "Inventory full"},
                )
            ],
            tags=["inventory"],
        )

        engine.add_rules([health_rule, inventory_rule])
        return engine

    @pytest.fixture
    def trigger_manager(self, rules_engine):
        """Create trigger manager with real rules engine."""
        return RuleTriggerManager(rules_engine)

    def test_process_health_change_event(self, trigger_manager):
        """Test processing health change event."""
        event_data = {
            "player_state": {"health": 20, "inventory_count": 5},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        results = trigger_manager.process_event("health_changed", event_data)

        # Should trigger health warning rule
        assert len(results) > 0
        triggered_results = [r for r in results if r.triggered]
        assert len(triggered_results) > 0

        # Check that trigger count increased
        health_triggers = trigger_manager.get_triggers_by_type(
            TriggerType.HEALTH_CHANGED
        )
        assert any(t.trigger_count > 0 for t in health_triggers)

    def test_process_action_performed_event(self, trigger_manager):
        """Test processing action performed event."""
        event_data = {
            "action": "take",
            "player_state": {"health": 50, "inventory_count": 10},
            "action_parameters": {"item": "sword"},
        }

        results = trigger_manager.process_event("action_performed", event_data)

        # Should trigger inventory full rule
        assert len(results) > 0
        triggered_results = [r for r in results if r.triggered]
        assert len(triggered_results) > 0

    def test_process_unsupported_event(self, trigger_manager):
        """Test processing unsupported event type."""
        event_data = {"custom_data": "test"}

        results = trigger_manager.process_event("unsupported_event", event_data)

        # Should return empty results for unsupported events
        assert len(results) == 0

    def test_trigger_with_tag_filtering(self, trigger_manager):
        """Test that triggers with tags only evaluate matching rules."""
        # Add specific health trigger that only evaluates health rules
        health_trigger = RuleTrigger(
            trigger_type=TriggerType.HEALTH_CHANGED,
            rule_tags=["health"],  # Only health rules
        )
        trigger_manager.add_trigger(health_trigger)

        event_data = {
            "player_state": {"health": 20, "inventory_count": 10},
            "action": "take",
        }

        results = trigger_manager.process_event("health_changed", event_data)

        # Should only trigger health rules, not inventory rules
        triggered_results = [r for r in results if r.triggered]
        for result in triggered_results:
            rule = trigger_manager.rules_engine.get_rule(result.rule_id)
            if rule:
                assert "health" in rule.tags

    def test_trigger_condition_filtering(self, trigger_manager):
        """Test trigger with custom condition."""

        def low_health_condition(event_data):
            health = event_data.get("player_state", {}).get("health", 100)
            return health < 25

        conditional_trigger = RuleTrigger(
            trigger_type=TriggerType.HEALTH_CHANGED,
            condition=low_health_condition,
            rule_tags=["health"],
        )
        trigger_manager.add_trigger(conditional_trigger)

        # Test with health above threshold
        high_health_data = {"player_state": {"health": 50}}
        results_high = trigger_manager.process_event("health_changed", high_health_data)

        # Test with health below threshold
        low_health_data = {"player_state": {"health": 20}}
        results_low = trigger_manager.process_event("health_changed", low_health_data)

        # The conditional trigger should have activated only for low health
        # (though default triggers might also activate)
        assert conditional_trigger.trigger_count > 0


class TestEventListeners:
    """Test event listener functionality."""

    @pytest.fixture
    def trigger_manager(self):
        """Create trigger manager for testing."""
        rules_engine = Mock(spec=RulesEngine)
        return RuleTriggerManager(rules_engine)

    def test_add_event_listener(self, trigger_manager):
        """Test adding event listener."""
        callback_calls = []

        def test_callback(event_data):
            callback_calls.append(event_data)

        trigger_manager.add_event_listener("test_event", test_callback)

        assert "test_event" in trigger_manager._event_listeners
        assert test_callback in trigger_manager._event_listeners["test_event"]

    def test_remove_event_listener(self, trigger_manager):
        """Test removing event listener."""

        def test_callback(event_data):
            pass

        trigger_manager.add_event_listener("test_event", test_callback)
        result = trigger_manager.remove_event_listener("test_event", test_callback)

        assert result is True
        assert len(trigger_manager._event_listeners.get("test_event", [])) == 0

    def test_remove_nonexistent_listener(self, trigger_manager):
        """Test removing nonexistent event listener."""

        def test_callback(event_data):
            pass

        result = trigger_manager.remove_event_listener("nonexistent", test_callback)
        assert result is False


class TestTriggerStatistics:
    """Test trigger manager statistics."""

    @pytest.fixture
    def trigger_manager(self):
        """Create trigger manager for testing."""
        rules_engine = Mock(spec=RulesEngine)
        return RuleTriggerManager(rules_engine)

    def test_get_statistics(self, trigger_manager):
        """Test getting trigger statistics."""
        # Add some custom triggers
        trigger1 = RuleTrigger(trigger_type=TriggerType.CUSTOM_EVENT)
        trigger2 = RuleTrigger(trigger_type=TriggerType.CUSTOM_EVENT)
        trigger2.enabled = False  # Disabled trigger

        trigger_manager.add_trigger(trigger1)
        trigger_manager.add_trigger(trigger2)

        # Add event listener
        def test_callback(event_data):
            pass

        trigger_manager.add_event_listener("test_event", test_callback)

        # Simulate some trigger activations
        trigger1.trigger_count = 5

        stats = trigger_manager.get_statistics()

        assert stats["total_triggers"] > 0
        assert stats["enabled_triggers"] > 0
        assert stats["disabled_triggers"] >= 1
        assert stats["total_activations"] >= 5
        assert "custom_event" in stats["trigger_type_distribution"]
        assert "test_event" in stats["event_listeners"]
        assert stats["event_listeners"]["test_event"] == 1


class TestEventTypeMapping:
    """Test event type to trigger type mapping."""

    @pytest.fixture
    def trigger_manager(self):
        """Create trigger manager for testing."""
        rules_engine = Mock(spec=RulesEngine)
        return RuleTriggerManager(rules_engine)

    def test_action_event_mapping(self, trigger_manager):
        """Test action event mapping."""
        # These should all map to ACTION_PERFORMED
        action_events = ["action_performed", "action_complete"]

        for event in action_events:
            trigger_type = trigger_manager._map_event_to_trigger_type(event)
            assert trigger_type == TriggerType.ACTION_PERFORMED

    def test_state_change_mapping(self, trigger_manager):
        """Test state change event mapping."""
        state_events = ["state_changed", "player_state_changed"]

        for event in state_events:
            trigger_type = trigger_manager._map_event_to_trigger_type(event)
            assert trigger_type == TriggerType.STATE_CHANGED

    def test_location_change_mapping(self, trigger_manager):
        """Test location change event mapping."""
        location_events = ["location_changed", "move"]

        for event in location_events:
            trigger_type = trigger_manager._map_event_to_trigger_type(event)
            assert trigger_type == TriggerType.LOCATION_CHANGED

    def test_inventory_change_mapping(self, trigger_manager):
        """Test inventory change event mapping."""
        inventory_events = ["inventory_changed", "item_taken", "item_dropped"]

        for event in inventory_events:
            trigger_type = trigger_manager._map_event_to_trigger_type(event)
            assert trigger_type == TriggerType.INVENTORY_CHANGED

    def test_conversation_mapping(self, trigger_manager):
        """Test conversation event mapping."""
        conversation_events = ["conversation_started", "talk"]

        for event in conversation_events:
            trigger_type = trigger_manager._map_event_to_trigger_type(event)
            assert trigger_type == TriggerType.CONVERSATION_STARTED

    def test_unknown_event_mapping(self, trigger_manager):
        """Test unknown event mapping returns None."""
        trigger_type = trigger_manager._map_event_to_trigger_type("unknown_event")
        assert trigger_type is None
