"""
Unit tests for ObjectSystemIntegration.

Tests coordination between all object systems and unified action processing.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.game_loop.core.objects.object_integration import (
    ObjectSystemEvent,
    ObjectSystemIntegration,
)


class TestObjectSystemIntegration:
    """Test cases for ObjectSystemIntegration functionality."""

    @pytest.fixture
    def integration_system(self):
        """Create integration system for testing."""
        # Create mock systems
        inventory_manager = Mock()
        inventory_manager.create_inventory = AsyncMock(return_value="test_inventory")
        inventory_manager.add_item = AsyncMock(
            return_value=(True, {"item_id": "test_item"})
        )
        inventory_manager.remove_item = AsyncMock(
            return_value=(True, {"item_id": "test_item"})
        )
        inventory_manager.move_item = AsyncMock(return_value=True)

        interaction_system = Mock()
        interaction_system.process_object_interaction = AsyncMock()
        interaction_system.execute_tool_interaction = AsyncMock()
        interaction_system.process_object_combination = AsyncMock()
        interaction_system.get_available_interactions = AsyncMock(return_value=[])

        condition_manager = Mock()
        condition_manager.get_condition_description = AsyncMock(
            return_value="excellent condition"
        )
        condition_manager.repair_object = AsyncMock(
            return_value=(True, {"repair_amount": 0.2})
        )

        container_manager = Mock()
        container_manager.organize_container_contents = AsyncMock(
            return_value={"changes_made": 5}
        )

        crafting_manager = Mock()
        crafting_manager.start_crafting_session = AsyncMock(
            return_value=(True, {"session_id": "test_session"})
        )
        crafting_manager.complete_crafting_session = AsyncMock(
            return_value=(True, {"success": True})
        )

        object_manager = Mock()
        physics_engine = Mock()

        integration = ObjectSystemIntegration(
            inventory_manager=inventory_manager,
            interaction_system=interaction_system,
            condition_manager=condition_manager,
            container_manager=container_manager,
            crafting_manager=crafting_manager,
            object_manager=object_manager,
            physics_engine=physics_engine,
        )

        return integration

    @pytest.mark.asyncio
    async def test_process_unified_action_craft_item(self, integration_system):
        """Test processing craft item action."""
        action_data = {
            "recipe_id": "basic_sword",
            "component_sources": {"iron_ingot": "player_inventory"},
        }

        result = await integration_system.process_unified_action(
            "craft_item", "player1", "", action_data  # No specific target for crafting
        )

        assert "success" in result
        assert "action_id" in result
        assert "processing_time" in result
        # Should have called crafting manager methods
        integration_system.crafting.start_crafting_session.assert_called()
        integration_system.crafting.complete_crafting_session.assert_called()

    @pytest.mark.asyncio
    async def test_process_unified_action_use_tool(self, integration_system):
        """Test processing use tool action."""
        action_data = {
            "tool_id": "hammer",
            "action": "pound",
            "context": {"player_id": "player1"},
        }

        result = await integration_system.process_unified_action(
            "use_tool", "player1", "nail", action_data
        )

        assert "success" in result
        assert "action_id" in result
        # Should have called interaction system
        integration_system.interactions.execute_tool_interaction.assert_called()

    @pytest.mark.asyncio
    async def test_process_unified_action_combine_objects(self, integration_system):
        """Test processing combine objects action."""
        action_data = {
            "primary_object": "wood_plank",
            "secondary_objects": ["nail", "screw"],
            "context": {"player_id": "player1"},
        }

        result = await integration_system.process_unified_action(
            "combine_objects", "player1", "wood_plank", action_data
        )

        assert "success" in result
        assert "action_id" in result
        # Should have called interaction system
        integration_system.interactions.process_object_combination.assert_called()

    @pytest.mark.asyncio
    async def test_process_unified_action_organize_container(self, integration_system):
        """Test processing organize container action."""
        action_data = {"organization_type": "category"}

        result = await integration_system.process_unified_action(
            "organize_container", "player1", "player_backpack", action_data
        )

        assert "success" in result
        assert "action_id" in result
        # Should have called container manager
        integration_system.containers.organize_container_contents.assert_called()

    @pytest.mark.asyncio
    async def test_process_unified_action_repair_object(self, integration_system):
        """Test processing repair object action."""
        action_data = {
            "repair_materials": ["repair_kit"],
            "repair_skill": 5,
            "repair_tools": ["hammer"],
        }

        result = await integration_system.process_unified_action(
            "repair_object", "player1", "damaged_sword", action_data
        )

        assert "success" in result
        assert "action_id" in result
        # Should have called condition manager
        integration_system.conditions.repair_object.assert_called()

    @pytest.mark.asyncio
    async def test_process_unified_action_transfer_items(self, integration_system):
        """Test processing transfer items action."""
        action_data = {
            "from_inventory": "player_inventory",
            "to_inventory": "chest_inventory",
            "item_id": "sword",
            "quantity": 1,
        }

        result = await integration_system.process_unified_action(
            "transfer_items", "player1", "", action_data
        )

        assert "success" in result
        assert "action_id" in result
        # Should have called inventory manager
        integration_system.inventory.move_item.assert_called()

    @pytest.mark.asyncio
    async def test_process_unified_action_validation_failure(self, integration_system):
        """Test action processing with validation failure."""
        # Empty action data should fail validation
        result = await integration_system.process_unified_action(
            "craft_item", "", "target", {}  # No actor ID
        )

        assert result["success"] is False
        assert "validation failed" in result["error"].lower()
        assert "details" in result

    @pytest.mark.asyncio
    async def test_get_object_comprehensive_status(self, integration_system):
        """Test getting comprehensive object status."""
        status = await integration_system.get_object_comprehensive_status(
            "test_sword", include_history=False
        )

        assert "object_id" in status
        assert status["object_id"] == "test_sword"
        assert "systems" in status
        assert "overall_status" in status
        assert "recommendations" in status

        # Should include status from all systems
        assert "condition" in status["systems"]
        assert "inventory" in status["systems"]
        assert "interactions" in status["systems"]
        assert "containers" in status["systems"]
        assert "crafting" in status["systems"]

    @pytest.mark.asyncio
    async def test_optimize_system_performance(self, integration_system):
        """Test system performance optimization."""
        optimization_targets = ["inventory", "interactions", "conditions"]

        result = await integration_system.optimize_system_performance(
            optimization_targets
        )

        assert result["optimization_complete"] is True
        assert result["targets"] == optimization_targets
        assert "results" in result
        assert "overall_improvement" in result

        # Should have optimization results for each target
        for target in optimization_targets:
            assert target in result["results"]

    @pytest.mark.asyncio
    async def test_optimize_system_performance_all_systems(self, integration_system):
        """Test optimizing all systems when no targets specified."""
        result = await integration_system.optimize_system_performance()

        assert result["optimization_complete"] is True
        assert len(result["targets"]) == 5  # All systems
        assert "inventory" in result["targets"]
        assert "interactions" in result["targets"]
        assert "conditions" in result["targets"]
        assert "containers" in result["targets"]
        assert "crafting" in result["targets"]

    @pytest.mark.asyncio
    async def test_synchronize_object_state(self, integration_system):
        """Test object state synchronization across systems."""
        result = await integration_system.synchronize_object_state("test_object")

        assert "object_id" in result
        assert result["object_id"] == "test_object"
        assert "systems_updated" in result
        assert "conflicts_resolved" in result
        assert "sync_successful" in result

    @pytest.mark.asyncio
    async def test_synchronize_object_state_force_update(self, integration_system):
        """Test forced object state synchronization."""
        result = await integration_system.synchronize_object_state(
            "test_object", force_update=True
        )

        assert result["sync_successful"] is True
        # Should update systems even if no conflicts detected
        assert len(result["systems_updated"]) >= 0

    @pytest.mark.asyncio
    async def test_process_batch_operations_parallel(self, integration_system):
        """Test processing batch operations in parallel."""
        operations = [
            {
                "action_type": "use_tool",
                "actor_id": "player1",
                "target_object": "nail",
                "action_data": {"tool_id": "hammer", "action": "pound"},
            },
            {
                "action_type": "transfer_items",
                "actor_id": "player1",
                "target_object": "",
                "action_data": {
                    "from_inventory": "inv1",
                    "to_inventory": "inv2",
                    "item_id": "sword",
                    "quantity": 1,
                },
            },
        ]

        results = await integration_system.process_batch_operations(
            operations, parallel=True
        )

        assert len(results) == 2
        for i, result in enumerate(results):
            assert "operation_index" in result
            assert result["operation_index"] == i
            assert "success" in result

    @pytest.mark.asyncio
    async def test_process_batch_operations_sequential(self, integration_system):
        """Test processing batch operations sequentially."""
        operations = [
            {
                "action_type": "repair_object",
                "actor_id": "player1",
                "target_object": "sword",
                "action_data": {
                    "repair_materials": ["repair_kit"],
                    "repair_skill": 5,
                    "repair_tools": ["hammer"],
                },
            }
        ]

        results = await integration_system.process_batch_operations(
            operations, parallel=False
        )

        assert len(results) == 1
        assert results[0]["operation_index"] == 0
        assert "success" in results[0]

    @pytest.mark.asyncio
    async def test_get_system_health_report(self, integration_system):
        """Test getting system health report."""
        health_report = await integration_system.get_system_health_report()

        assert "timestamp" in health_report
        assert "overall_health" in health_report
        assert "systems" in health_report
        assert "performance_metrics" in health_report
        assert "recommendations" in health_report

        # Should check all systems
        systems = health_report["systems"]
        assert "inventory" in systems
        assert "interactions" in systems
        assert "conditions" in systems
        assert "containers" in systems
        assert "crafting" in systems

        # Each system should have health information
        for system_name, system_health in systems.items():
            assert "status" in system_health
            assert "initialized" in system_health

    @pytest.mark.asyncio
    async def test_validate_action_prerequisites_success(self, integration_system):
        """Test successful action prerequisite validation."""
        is_valid, errors = await integration_system.validate_action_prerequisites(
            "use_tool", "player1", "target_object", {"tool_id": "hammer"}
        )

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_action_prerequisites_missing_actor(
        self, integration_system
    ):
        """Test validation failure due to missing actor."""
        is_valid, errors = await integration_system.validate_action_prerequisites(
            "use_tool", "", "target_object", {"tool_id": "hammer"}  # No actor ID
        )

        assert is_valid is False
        assert any("actor id" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_action_prerequisites_craft_item_no_recipe(
        self, integration_system
    ):
        """Test validation failure for craft item without recipe or components."""
        is_valid, errors = await integration_system.validate_action_prerequisites(
            "craft_item", "player1", "", {}  # No recipe_id or components
        )

        assert is_valid is False
        assert any(
            "recipe" in error.lower() or "components" in error.lower()
            for error in errors
        )

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, integration_system):
        """Test that performance metrics are tracked."""
        # Process an action to generate metrics
        await integration_system.process_unified_action(
            "transfer_items",
            "player1",
            "",
            {
                "from_inventory": "inv1",
                "to_inventory": "inv2",
                "item_id": "item",
                "quantity": 1,
            },
        )

        # Check that metrics were recorded
        metrics = integration_system._performance_metrics
        assert "transfer_items" in metrics

        action_metrics = metrics["transfer_items"]
        assert "total_executions" in action_metrics
        assert "successful_executions" in action_metrics
        assert "total_time" in action_metrics
        assert "average_time" in action_metrics
        assert "success_rate" in action_metrics
        assert action_metrics["total_executions"] >= 1

    @pytest.mark.asyncio
    async def test_cross_system_effects_application(self, integration_system):
        """Test that cross-system effects are applied."""
        # Mock interaction result with tool usage
        mock_interaction_result = Mock()
        mock_interaction_result.tool_used = "hammer"
        mock_interaction_result.success = True

        # Process action that should trigger cross-system effects
        await integration_system._apply_cross_system_effects(
            "use_tool",
            "player1",
            "nail",
            {"success": True, "interaction_result": mock_interaction_result},
        )

        # Should call wear and degradation on the tool
        integration_system.interactions.apply_wear_and_degradation.assert_called()

    @pytest.mark.asyncio
    async def test_event_system_initialization(self, integration_system):
        """Test that event system is properly initialized."""
        # Check that event handlers are set up
        assert "object_condition_changed" in integration_system._event_handlers
        assert "inventory_updated" in integration_system._event_handlers
        assert "crafting_completed" in integration_system._event_handlers

    @pytest.mark.asyncio
    async def test_handle_condition_change_event(self, integration_system):
        """Test handling of condition change events."""
        event = ObjectSystemEvent(
            event_type="object_condition_changed",
            source_system="conditions",
            object_id="test_sword",
            data={"new_condition": 0.05},  # Severely damaged
            timestamp=asyncio.get_event_loop().time(),
        )

        # Should handle without error
        await integration_system._handle_condition_change(event)

    @pytest.mark.asyncio
    async def test_error_handling_invalid_action_type(self, integration_system):
        """Test error handling for invalid action types."""
        result = await integration_system.process_unified_action(
            "invalid_action_type", "player1", "target", {}
        )

        # Should handle gracefully and process as generic action
        assert "success" in result
        assert "action_id" in result

    @pytest.mark.asyncio
    async def test_object_status_assessment(self, integration_system):
        """Test object status assessment logic."""
        # Test with systems that have errors
        systems_with_errors = {
            "condition": {"error": "System unavailable"},
            "inventory": {"location": "chest", "accessible": True},
            "interactions": {"available_count": 3},
        }

        status = integration_system._assess_overall_object_status(systems_with_errors)
        assert status == "error"

        # Test with systems needing attention
        systems_needing_attention = {
            "condition": {"needs_attention": True},
            "inventory": {"location": "chest", "accessible": True},
        }

        status = integration_system._assess_overall_object_status(
            systems_needing_attention
        )
        assert status == "needs_attention"

        # Test with good systems
        good_systems = {
            "condition": {"description": "excellent"},
            "inventory": {"location": "chest", "accessible": True},
        }

        status = integration_system._assess_overall_object_status(good_systems)
        assert status == "good"

    @pytest.mark.asyncio
    async def test_generate_object_recommendations(self, integration_system):
        """Test object recommendation generation."""
        systems_status = {
            "condition": {"needs_attention": True},
            "interactions": {"interactions": []},  # No available interactions
        }

        recommendations = integration_system._generate_object_recommendations(
            systems_status
        )

        assert len(recommendations) >= 1
        assert any(
            "repair" in rec.lower() or "maintenance" in rec.lower()
            for rec in recommendations
        )
        assert any("interaction" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_health_report_generation(self, integration_system):
        """Test health report generation logic."""
        health_report = await integration_system.get_system_health_report()

        # Should categorize overall health
        assert health_report["overall_health"] in [
            "excellent",
            "good",
            "fair",
            "poor",
            "error",
        ]

        # Should generate recommendations if needed
        if health_report["overall_health"] in ["fair", "poor"]:
            assert len(health_report["recommendations"]) > 0
