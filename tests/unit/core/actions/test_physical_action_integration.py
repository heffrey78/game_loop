"""
Unit tests for PhysicalActionIntegration.

This module tests the physical action integration functionality including
multi-step coordination, interruption handling, and performance optimization.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from game_loop.core.actions.physical_action_integration import (
    ActionExecutionContext,
    PhysicalActionIntegration,
)
from game_loop.core.actions.types import ActionClassification, ActionType
from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionResult,
    PhysicalActionType,
)


class TestPhysicalActionIntegration:
    """Test cases for PhysicalActionIntegration."""

    @pytest.fixture
    def mock_physical_processor(self):
        """Fixture for mock physical action processor."""
        mock_processor = Mock()
        mock_processor.process_physical_action = AsyncMock(
            return_value=PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=["test_entity"],
                state_changes={"test_change": True},
                energy_cost=10.0,
                time_elapsed=5.0,
                side_effects=[],
                description="Test action completed successfully.",
            )
        )
        mock_processor.validate_action_feasibility = AsyncMock(
            return_value=(True, None)
        )
        return mock_processor

    @pytest.fixture
    def mock_movement_manager(self):
        """Fixture for mock movement manager."""
        mock_movement = Mock()
        mock_movement.process_movement_command = AsyncMock(
            return_value=PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MOVEMENT,
                affected_entities=["player"],
                state_changes={"player_location": "new_location"},
                energy_cost=8.0,
                time_elapsed=3.0,
                side_effects=[],
                description="Movement completed successfully.",
            )
        )
        return mock_movement

    @pytest.fixture
    def mock_environment_manager(self):
        """Fixture for mock environment interaction manager."""
        mock_environment = Mock()
        mock_environment.process_environment_interaction = AsyncMock(
            return_value=PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=["container"],
                state_changes={"container_open": True},
                energy_cost=5.0,
                time_elapsed=2.0,
                side_effects=[],
                description="Container interaction completed.",
            )
        )
        return mock_environment

    @pytest.fixture
    def mock_physics_engine(self):
        """Fixture for mock physics engine."""
        mock_physics = Mock()
        mock_physics.validate_physical_constraints = AsyncMock(
            return_value=(True, None)
        )
        return mock_physics

    @pytest.fixture
    def mock_spatial_navigator(self):
        """Fixture for mock spatial navigator."""
        return Mock()

    @pytest.fixture
    def mock_game_state_manager(self):
        """Fixture for mock game state manager."""
        return Mock()

    @pytest.fixture
    def integration_layer(
        self,
        mock_physical_processor,
        mock_movement_manager,
        mock_environment_manager,
        mock_physics_engine,
        mock_spatial_navigator,
        mock_game_state_manager,
    ):
        """Fixture for PhysicalActionIntegration instance."""
        return PhysicalActionIntegration(
            physical_processor=mock_physical_processor,
            movement_manager=mock_movement_manager,
            environment_manager=mock_environment_manager,
            physics_engine=mock_physics_engine,
            spatial_navigator=mock_spatial_navigator,
            game_state_manager=mock_game_state_manager,
        )

    @pytest.fixture
    def sample_action_classification(self):
        """Fixture for sample action classification."""
        return ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="wooden_crate",
            intent="push the wooden crate to the corner",
            raw_input="push crate",
            parameters={"direction": "north", "force": "moderate"},
            secondary_targets=["corner"],
        )

    @pytest.fixture
    def sample_context(self):
        """Fixture for sample action context."""
        return {
            "player_id": "player_1",
            "player_state": {
                "current_location": "warehouse",
                "energy": 85,
                "strength": 55,
                "skills": {"athletics": 4, "strength": 6},
            },
            "current_location": "warehouse",
        }

    @pytest.mark.asyncio
    async def test_process_classified_physical_action_success(
        self, integration_layer, sample_action_classification, sample_context
    ):
        """Test successful classified physical action processing."""
        result = await integration_layer.process_classified_physical_action(
            sample_action_classification, sample_context
        )

        assert isinstance(result, dict)
        assert "success" in result
        assert "action_type" in result
        assert "classification" in result

    @pytest.mark.asyncio
    async def test_process_classified_physical_action_feasibility_failure(
        self, integration_layer
    ):
        """Test classified physical action with feasibility failure."""
        # Mock feasibility check to fail
        integration_layer.validate_action_chain_feasibility = AsyncMock(
            return_value=(False, 0)  # Fails at step 0
        )

        action = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.8,
            target="heavy_boulder",
            intent="lift the boulder",
        )

        context = {
            "player_id": "weak_player",
            "player_state": {"energy": 5, "strength": 10},
        }

        result = await integration_layer.process_classified_physical_action(
            action, context
        )

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_coordinate_multi_step_actions_success(
        self, integration_layer, sample_context
    ):
        """Test successful multi-step action coordination."""
        action_sequence = [
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="door",
                    intent="open door",
                ),
                "context": sample_context,
            },
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.8,
                    target="north",
                    intent="walk north",
                ),
                "context": sample_context,
            },
        ]

        results = await integration_layer.coordinate_multi_step_actions(
            action_sequence, sample_context
        )

        assert isinstance(results, list)
        assert len(results) == len(action_sequence)
        for result in results:
            assert isinstance(result, PhysicalActionResult)

    @pytest.mark.asyncio
    async def test_coordinate_multi_step_actions_with_failure(
        self, integration_layer, sample_context
    ):
        """Test multi-step action coordination with step failure."""
        # Configure first action to succeed, second to fail
        failing_processor = Mock()
        failing_processor.process_physical_action = AsyncMock(
            side_effect=[
                PhysicalActionResult(
                    success=True,
                    action_type=PhysicalActionType.OPENING,
                    affected_entities=["door"],
                    state_changes={"door_open": True},
                    energy_cost=5.0,
                    time_elapsed=2.0,
                    side_effects=[],
                    description="Door opened.",
                ),
                PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=["player"],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Movement failed.",
                    error_message="Path blocked",
                ),
            ]
        )

        failing_processor.validate_action_feasibility = AsyncMock(
            return_value=(True, None)
        )
        integration_layer.execution_context.physical_processor = failing_processor

        action_sequence = [
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL, confidence=0.9, target="door"
                )
            },
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL, confidence=0.9, target="north"
                )
            },
        ]

        results = await integration_layer.coordinate_multi_step_actions(
            action_sequence, sample_context
        )

        assert isinstance(results, list)
        assert len(results) >= 1  # At least first action should complete

    @pytest.mark.asyncio
    async def test_handle_action_interruptions_continue(
        self, integration_layer, sample_context
    ):
        """Test action interruption handling when action can continue."""
        active_action = {
            "classification": ActionClassification(
                action_type=ActionType.PHYSICAL, confidence=0.9, target="climb_rope"
            ),
            "start_time": 1000.0,
            "progress": 0.6,
        }

        interruption_event = {
            "type": "distraction",
            "severity": "low",
            "description": "A bird flies by",
        }

        result = await integration_layer.handle_action_interruptions(
            active_action, interruption_event
        )

        assert isinstance(result, dict)
        assert "action_status" in result
        assert "interruption_handled" in result

    @pytest.mark.asyncio
    async def test_handle_action_interruptions_stop(
        self, integration_layer, sample_context
    ):
        """Test action interruption handling when action must stop."""
        active_action = {
            "classification": ActionClassification(
                action_type=ActionType.PHYSICAL, confidence=0.9, target="heavy_lift"
            ),
            "start_time": 1000.0,
            "progress": 0.3,
        }

        critical_interruption = {
            "type": "structural_failure",
            "severity": "critical",
            "description": "Support beam breaks",
        }

        result = await integration_layer.handle_action_interruptions(
            active_action, critical_interruption
        )

        assert isinstance(result, dict)
        assert "action_status" in result

    @pytest.mark.asyncio
    async def test_optimize_action_performance(self, integration_layer):
        """Test action performance optimization."""
        frequency_data = {
            "usage_count": 150,
            "avg_execution_time": 6.0,
            "success_rate": 0.75,
        }

        await integration_layer.optimize_action_performance(
            PhysicalActionType.PUSHING, frequency_data
        )

        # Should complete without errors

    @pytest.mark.asyncio
    async def test_validate_action_chain_feasibility_success(
        self, integration_layer, sample_context
    ):
        """Test successful action chain feasibility validation."""
        action_chain = [
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="lever",
                    intent="pull lever",
                ),
                "context": sample_context,
            },
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="door",
                    intent="walk through door",
                ),
                "context": sample_context,
            },
        ]

        is_feasible, failing_step = (
            await integration_layer.validate_action_chain_feasibility(
                action_chain, sample_context
            )
        )

        assert isinstance(is_feasible, bool)
        if not is_feasible:
            assert isinstance(failing_step, int)

    @pytest.mark.asyncio
    async def test_apply_learning_effects(self, integration_layer):
        """Test applying learning effects from repeated actions."""
        await integration_layer.apply_learning_effects(
            "player_1", PhysicalActionType.CLIMBING, 0.85
        )

        # Check that metrics were updated
        assert "player_1" in integration_layer._action_metrics
        player_metrics = integration_layer._action_metrics["player_1"]
        assert "climbing" in player_metrics

    @pytest.mark.asyncio
    async def test_handle_concurrent_actions_success(
        self, integration_layer, sample_context
    ):
        """Test successful concurrent action handling."""
        concurrent_actions = [
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="left_hand_action",
                    intent="use left hand",
                ),
                "context": sample_context,
            },
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="right_hand_action",
                    intent="use right hand",
                ),
                "context": sample_context,
            },
        ]

        results = await integration_layer.handle_concurrent_actions(
            concurrent_actions, sample_context
        )

        assert isinstance(results, list)
        assert len(results) == len(concurrent_actions)
        for result in results:
            assert isinstance(result, PhysicalActionResult)

    @pytest.mark.asyncio
    async def test_handle_concurrent_actions_with_conflicts(
        self, integration_layer, sample_context
    ):
        """Test concurrent action handling with conflicts."""
        # Mock conflict detection
        integration_layer._detect_action_conflicts = AsyncMock(
            return_value=[{"type": "resource_conflict", "actions": [0, 1]}]
        )
        integration_layer._resolve_action_conflicts = AsyncMock(
            return_value=[
                {
                    "classification": ActionClassification(
                        action_type=ActionType.PHYSICAL,
                        confidence=0.9,
                        target="resolved",
                    )
                }
            ]
        )

        conflicting_actions = [
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="same_target",
                )
            },
            {
                "classification": ActionClassification(
                    action_type=ActionType.PHYSICAL,
                    confidence=0.9,
                    target="same_target",
                )
            },
        ]

        results = await integration_layer.handle_concurrent_actions(
            conflicting_actions, sample_context
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_generate_action_feedback(self, integration_layer, sample_context):
        """Test action feedback generation."""
        action_result = PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.CLIMBING,
            affected_entities=["rope", "cliff"],
            state_changes={"player_height": 10},
            energy_cost=25.0,  # High energy cost
            time_elapsed=15.0,  # Long time
            side_effects=["rope_wear", "dust_cloud"],
            description="You climb the rope to reach the cliff ledge.",
        )

        feedback = await integration_layer.generate_action_feedback(
            action_result, sample_context
        )

        assert isinstance(feedback, str)
        assert len(feedback) > 0
        # Should enhance the basic description
        assert "climb" in feedback.lower()

    @pytest.mark.asyncio
    async def test_update_world_physics_state(self, integration_layer):
        """Test world physics state updating."""
        action_results = [
            PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.BREAKING,
                affected_entities=["wooden_crate"],
                state_changes={"crate_broken": True},
                energy_cost=15.0,
                time_elapsed=8.0,
                side_effects=["debris"],
                description="Crate broken into pieces.",
            ),
        ]

        await integration_layer.update_world_physics_state(action_results)

        # Should complete without errors

    def test_register_action_integration_hook(self, integration_layer):
        """Test registering action integration hooks."""
        mock_hook = Mock()

        integration_layer.register_action_integration_hook("pre_execution", mock_hook)

        assert mock_hook in integration_layer._integration_hooks["pre_execution"]

    def test_register_invalid_hook_type(self, integration_layer):
        """Test registering hook with invalid type."""
        mock_hook = Mock()

        # Should handle invalid hook type gracefully
        integration_layer.register_action_integration_hook("invalid_hook", mock_hook)

    @pytest.mark.asyncio
    async def test_execution_strategy_determination(
        self, integration_layer, sample_context
    ):
        """Test execution strategy determination."""
        # Test movement strategy
        movement_action = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="north",
            intent="move north",
        )

        strategy = await integration_layer._determine_execution_strategy(
            movement_action, sample_context
        )
        assert strategy == "movement"

        # Test environment strategy
        environment_action = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="container",
            intent="open the container",
        )

        strategy = await integration_layer._determine_execution_strategy(
            environment_action, sample_context
        )
        assert strategy == "environment"

    @pytest.mark.asyncio
    async def test_movement_action_execution(self, integration_layer, sample_context):
        """Test movement-specific action execution."""
        movement_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="east",
            intent="go east",
        )

        result = await integration_layer._execute_movement_action(
            movement_classification, sample_context
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_environment_action_execution(
        self, integration_layer, sample_context
    ):
        """Test environment-specific action execution."""
        environment_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="chest",
            intent="open chest",
        )

        result = await integration_layer._execute_environment_action(
            environment_classification, sample_context
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_complex_action_execution(self, integration_layer, sample_context):
        """Test complex multi-component action execution."""
        complex_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="mechanism",
            intent="operate complex mechanism",
            secondary_targets=["lever1", "lever2", "button"],
        )

        result = await integration_layer._execute_complex_action(
            complex_classification, sample_context
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_action_result_conversion(self, integration_layer):
        """Test action result conversion between formats."""
        action_result = PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=["test_entity"],
            state_changes={"test": True},
            energy_cost=10.0,
            time_elapsed=5.0,
            side_effects=["side_effect"],
            description="Test description",
        )

        # Convert to dict
        result_dict = await integration_layer._convert_action_result_to_dict(
            action_result
        )
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["action_type"] in [
            "manipulation",
            PhysicalActionType.MANIPULATION,
        ]

        # Convert back to result
        converted_result = await integration_layer._convert_dict_to_action_result(
            result_dict
        )
        assert isinstance(converted_result, PhysicalActionResult)
        assert converted_result.success is True

    @pytest.mark.asyncio
    async def test_action_type_extraction(self, integration_layer):
        """Test physical action type extraction from classification."""
        movement_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="north",
            intent="move north",
        )

        action_type = await integration_layer._extract_physical_action_type(
            movement_classification
        )
        assert action_type == PhysicalActionType.MOVEMENT

        push_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="boulder",
            intent="push the boulder",
        )

        action_type = await integration_layer._extract_physical_action_type(
            push_classification
        )
        assert action_type == PhysicalActionType.PUSHING

    @pytest.mark.asyncio
    async def test_action_execution_simulation(self, integration_layer, sample_context):
        """Test action execution simulation for planning."""
        classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="test_object",
            intent="test action",
        )

        simulation_result = await integration_layer._simulate_action_execution(
            classification, sample_context
        )

        assert isinstance(simulation_result, dict)
        assert "state_changes" in simulation_result

    @pytest.mark.asyncio
    async def test_action_metrics_update(
        self, integration_layer, sample_action_classification
    ):
        """Test action metrics updating."""
        result = {"success": True, "time_elapsed": 8.0}

        await integration_layer._update_integration_metrics(
            sample_action_classification, result
        )

        action_type = sample_action_classification.action_type.value
        assert action_type in integration_layer._action_metrics
        metrics = integration_layer._action_metrics[action_type]
        assert metrics["total_executions"] >= 1

    @pytest.mark.asyncio
    async def test_error_handling(self, integration_layer):
        """Test error handling in integration operations."""
        # Test with invalid classification
        invalid_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target=None,
            intent=None,
        )

        # Override the mock to simulate validation failure
        integration_layer.execution_context.physical_processor.validate_action_feasibility = AsyncMock(
            return_value=(False, "No player context provided")
        )

        # Test with empty context that should cause validation failure
        result = await integration_layer.process_classified_physical_action(
            invalid_classification, {}  # No player_id or player_state
        )

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_hook_execution(
        self, integration_layer, sample_action_classification, sample_context
    ):
        """Test integration hook execution."""
        # Register test hooks
        pre_hook = AsyncMock(return_value={"hook_executed": True})
        post_hook = AsyncMock(return_value={"post_hook_executed": True})

        integration_layer.register_action_integration_hook("pre_execution", pre_hook)
        integration_layer.register_action_integration_hook("post_execution", post_hook)

        # Execute action to trigger hooks
        result = await integration_layer.process_classified_physical_action(
            sample_action_classification, sample_context
        )

        # Verify hooks were called
        pre_hook.assert_called_once()
        post_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_effects_progression(self, integration_layer):
        """Test learning effects progression over multiple actions."""
        player_id = "learning_player"
        action_type = PhysicalActionType.CLIMBING

        # Apply learning effects multiple times
        for i in range(5):
            await integration_layer.apply_learning_effects(player_id, action_type, 0.8)

        # Check that experience and skill accumulated
        player_metrics = integration_layer._action_metrics[player_id]
        climbing_data = player_metrics["climbing"]
        assert climbing_data["attempts"] == 5
        assert climbing_data["experience"] > 0


class TestActionExecutionContext:
    """Test cases for ActionExecutionContext."""

    def test_action_execution_context_creation(self):
        """Test ActionExecutionContext creation."""
        mock_components = [Mock() for _ in range(6)]

        context = ActionExecutionContext(*mock_components)

        assert context.physical_processor == mock_components[0]
        assert context.movement == mock_components[1]
        assert context.environment == mock_components[2]
        assert context.physics == mock_components[3]
        assert context.navigator == mock_components[4]
        assert context.state == mock_components[5]


@pytest.mark.asyncio
async def test_integration_full_workflow():
    """Test PhysicalActionIntegration full workflow integration."""
    # Create all mocks
    mock_physical = Mock()
    mock_physical.process_physical_action = AsyncMock(
        return_value=PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=["test"],
            state_changes={},
            energy_cost=10.0,
            time_elapsed=5.0,
            side_effects=[],
            description="Success",
        )
    )
    mock_physical.validate_action_feasibility = AsyncMock(return_value=(True, None))

    mock_movement = Mock()
    mock_environment = Mock()
    mock_physics = Mock()
    mock_navigator = Mock()
    mock_state = Mock()

    # Create integration layer
    integration = PhysicalActionIntegration(
        mock_physical,
        mock_movement,
        mock_environment,
        mock_physics,
        mock_navigator,
        mock_state,
    )

    # Test complex action workflow
    action = ActionClassification(
        action_type=ActionType.PHYSICAL,
        confidence=0.95,
        target="complex_mechanism",
        intent="operate the complex mechanism",
        parameters={"approach": "careful", "tools": ["wrench", "oil"]},
    )

    context = {
        "player_id": "master_mechanic",
        "player_state": {
            "current_location": "engine_room",
            "energy": 90,
            "strength": 70,
            "skills": {"mechanics": 8, "engineering": 6},
            "inventory": ["wrench", "oil", "flashlight"],
        },
    }

    # Process action
    result = await integration.process_classified_physical_action(action, context)

    # Verify result
    assert isinstance(result, dict)
    assert "success" in result
    assert "action_type" in result

    # Test learning effects
    await integration.apply_learning_effects(
        "master_mechanic", PhysicalActionType.MANIPULATION, 0.9
    )

    # Verify learning was recorded
    assert "master_mechanic" in integration._action_metrics


if __name__ == "__main__":
    pytest.main([__file__])
