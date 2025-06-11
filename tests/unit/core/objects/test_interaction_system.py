"""
Unit tests for ObjectInteractionSystem.

Tests complex object interactions, tool usage, and state transitions.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.game_loop.core.objects.interaction_system import (
    InteractionResult,
    ObjectInteractionSystem,
    ObjectInteractionType,
)


class TestObjectInteractionSystem:
    """Test cases for ObjectInteractionSystem functionality."""

    @pytest.fixture
    def interaction_system(self):
        """Create interaction system for testing."""
        object_manager = Mock()
        object_manager.get_object_properties = AsyncMock(
            return_value={
                "name": "Test Object",
                "usable": True,
                "consumable": False,
                "disassemblable": False,
                "durability": 1.0,
                "wear_resistance": 1.0,
                "compatible_actions": ["use", "examine", "cut"],
            }
        )
        object_manager.object_exists = AsyncMock(return_value=True)

        physics_engine = Mock()
        skill_manager = Mock()
        recipe_manager = Mock()

        system = ObjectInteractionSystem(
            object_manager, physics_engine, skill_manager, recipe_manager
        )
        return system

    @pytest.mark.asyncio
    async def test_process_examine_interaction(self, interaction_system):
        """Test examine interaction processing."""
        result = await interaction_system.process_object_interaction(
            ObjectInteractionType.EXAMINE,
            "test_object",
            None,
            None,
            {"player_state": {"skills": {}}},
        )

        assert isinstance(result, InteractionResult)
        assert result.success is True
        assert result.interaction_type == ObjectInteractionType.EXAMINE
        assert result.source_object == "test_object"
        assert "examine" in result.description.lower()

    @pytest.mark.asyncio
    async def test_process_use_interaction(self, interaction_system):
        """Test use interaction processing."""
        result = await interaction_system.process_object_interaction(
            ObjectInteractionType.USE,
            "test_object",
            None,
            None,
            {"player_state": {"skills": {"tool_use": 5}}},
        )

        assert isinstance(result, InteractionResult)
        assert result.interaction_type == ObjectInteractionType.USE
        assert result.source_object == "test_object"

    @pytest.mark.asyncio
    async def test_validate_interaction_requirements_success(self, interaction_system):
        """Test successful interaction requirement validation."""
        is_valid, errors = await interaction_system.validate_interaction_requirements(
            ObjectInteractionType.USE,
            ["test_object"],
            {"skills": {"tool_use": 5}, "energy": 100},
        )

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_interaction_requirements_insufficient_energy(
        self, interaction_system
    ):
        """Test validation failure due to insufficient energy."""
        is_valid, errors = await interaction_system.validate_interaction_requirements(
            ObjectInteractionType.REPAIR,
            ["test_object"],
            {"skills": {"repair": 5}, "energy": 1},  # Very low energy
        )

        assert is_valid is False
        assert any("energy" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_interaction_requirements_missing_skills(
        self, interaction_system
    ):
        """Test validation failure due to missing skills."""
        is_valid, errors = await interaction_system.validate_interaction_requirements(
            ObjectInteractionType.REPAIR,
            ["test_object"],
            {"skills": {}, "energy": 100},  # No repair skill
        )

        assert is_valid is False
        assert any("repair" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_get_available_interactions(self, interaction_system):
        """Test getting available interactions for an object."""
        interactions = await interaction_system.get_available_interactions(
            "test_object", {"player_id": "test_player"}
        )

        assert len(interactions) > 0

        # Should always have examine
        examine_interaction = next(
            (i for i in interactions if i["type"] == ObjectInteractionType.EXAMINE),
            None,
        )
        assert examine_interaction is not None

        # Should have use since object is usable
        use_interaction = next(
            (i for i in interactions if i["type"] == ObjectInteractionType.USE), None
        )
        assert use_interaction is not None

    @pytest.mark.asyncio
    async def test_execute_tool_interaction_success(self, interaction_system):
        """Test successful tool interaction execution."""
        result = await interaction_system.execute_tool_interaction(
            "hammer",
            "nail",
            "cut",
            {"player_id": "test_player", "player_state": {"skills": {"tool_use": 5}}},
        )

        assert isinstance(result, InteractionResult)
        assert result.tool_used == "hammer"
        assert result.source_object == "nail"

    @pytest.mark.asyncio
    async def test_execute_tool_interaction_incompatible(self, interaction_system):
        """Test tool interaction with incompatible action."""
        # Mock tool with limited compatible actions
        interaction_system.objects.get_object_properties = AsyncMock(
            side_effect=lambda obj_id: {
                "hammer": {
                    "name": "Hammer",
                    "compatible_actions": ["pound"],  # Does not include "cut"
                },
                "nail": {"name": "Nail", "usable": True},
            }.get(obj_id, {})
        )

        result = await interaction_system.execute_tool_interaction(
            "hammer", "nail", "cut", {"player_id": "test_player"}  # Incompatible action
        )

        assert result.success is False
        assert "incompatible" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_process_object_combination(self, interaction_system):
        """Test object combination processing."""
        result = await interaction_system.process_object_combination(
            "wood_plank",
            ["nail", "screw"],
            None,  # No specific recipe
            {"player_id": "test_player", "player_state": {"skills": {"crafting": 3}}},
        )

        assert isinstance(result, InteractionResult)
        assert result.interaction_type == ObjectInteractionType.COMBINE
        assert result.source_object == "wood_plank"

    @pytest.mark.asyncio
    async def test_handle_object_state_transition(self, interaction_system):
        """Test object state transition handling."""
        result = await interaction_system.handle_object_state_transition(
            "test_object", "activate", {"player_id": "test_player"}
        )

        # Should create default state machine if none exists
        assert "test_object" in interaction_system._state_machines

        # Check transition result structure
        assert "success" in result
        assert "from_state" in result or "error" in result

    @pytest.mark.asyncio
    async def test_calculate_interaction_success_probability(self, interaction_system):
        """Test interaction success probability calculation."""
        probability = (
            await interaction_system.calculate_interaction_success_probability(
                ObjectInteractionType.USE, ["test_object"], {"tool_use": 5}
            )
        )

        assert 0.0 <= probability <= 1.0

        # USE should have high base probability
        assert probability > 0.8

    @pytest.mark.asyncio
    async def test_apply_wear_and_degradation(self, interaction_system):
        """Test wear and degradation application."""
        result = await interaction_system.apply_wear_and_degradation(
            "test_tool", 1.0, ObjectInteractionType.USE  # Normal usage intensity
        )

        assert "object_id" in result
        assert "wear_applied" in result
        assert "old_condition" in result
        assert "new_condition" in result
        assert result["object_id"] == "test_tool"

    @pytest.mark.asyncio
    async def test_register_interaction_handler(self, interaction_system):
        """Test registering custom interaction handlers."""

        # Create mock handler
        async def custom_handler(source, target, tool, context, success):
            return {"custom": "result"}

        # Register handler
        interaction_system.register_interaction_handler(
            ObjectInteractionType.TRANSFORM, custom_handler, priority=10
        )

        # Verify handler is registered
        assert (
            ObjectInteractionType.TRANSFORM in interaction_system._interaction_handlers
        )
        handlers = interaction_system._interaction_handlers[
            ObjectInteractionType.TRANSFORM
        ]
        assert len(handlers) > 0
        assert handlers[0][0] == 10  # Priority
        assert handlers[0][1] == custom_handler

    @pytest.mark.asyncio
    async def test_discover_new_interactions(self, interaction_system):
        """Test discovery of new interaction possibilities."""
        # Mock high compatibility between objects
        interaction_system._check_object_compatibility = AsyncMock(return_value=0.8)

        discoveries = await interaction_system.discover_new_interactions(
            ["object1", "object2", "object3"], {"player_id": "test_player"}
        )

        # Should find some combinations with high compatibility
        assert len(discoveries) > 0
        for discovery in discoveries:
            assert "objects" in discovery
            assert "interaction_type" in discovery
            assert "compatibility" in discovery
            assert discovery["compatibility"] > 0.7

    @pytest.mark.asyncio
    async def test_interaction_with_consumable_object(self, interaction_system):
        """Test interaction with consumable objects."""
        # Mock consumable object
        interaction_system.objects.get_object_properties = AsyncMock(
            return_value={
                "name": "Health Potion",
                "usable": True,
                "consumable": True,
                "description": "Restores health when consumed",
            }
        )

        # Get available interactions
        interactions = await interaction_system.get_available_interactions(
            "health_potion", {"player_id": "test_player"}
        )

        # Should include consume interaction
        consume_interaction = next(
            (i for i in interactions if i["type"] == ObjectInteractionType.CONSUME),
            None,
        )
        assert consume_interaction is not None

    @pytest.mark.asyncio
    async def test_interaction_history_tracking(self, interaction_system):
        """Test that interactions are tracked in history."""
        # Process an interaction
        await interaction_system.process_object_interaction(
            ObjectInteractionType.USE,
            "test_object",
            None,
            None,
            {"player_id": "test_player", "player_state": {"skills": {}}},
        )

        # Check that history was recorded
        assert "test_player" in interaction_system._interaction_history
        history = interaction_system._interaction_history["test_player"]
        assert len(history) > 0
        assert "interaction_id" in history[0]
        assert "timestamp" in history[0]
        assert "result" in history[0]

    @pytest.mark.asyncio
    async def test_skill_experience_calculation(self, interaction_system):
        """Test skill experience calculation for successful interactions."""
        # Mock successful interaction
        experience = await interaction_system._calculate_skill_experience(
            ObjectInteractionType.REPAIR,
            True,  # Successful
            {"player_id": "test_player"},
        )

        # Should award repair experience for successful repair
        assert "repair" in experience
        assert experience["repair"] > 0

    @pytest.mark.asyncio
    async def test_energy_cost_calculation(self, interaction_system):
        """Test energy cost calculation for different interactions."""
        # Test different interaction types
        examine_cost = await interaction_system._calculate_energy_cost(
            ObjectInteractionType.EXAMINE, "test_object"
        )
        repair_cost = await interaction_system._calculate_energy_cost(
            ObjectInteractionType.REPAIR, "test_object"
        )

        # Examine should cost less than repair
        assert examine_cost < repair_cost
        assert examine_cost > 0
        assert repair_cost > 0

    @pytest.mark.asyncio
    async def test_object_condition_impact_on_success(self, interaction_system):
        """Test that object condition affects interaction success."""
        # Mock object with poor condition
        interaction_system._get_object_condition = AsyncMock(
            return_value=0.2
        )  # Poor condition

        probability = (
            await interaction_system.calculate_interaction_success_probability(
                ObjectInteractionType.USE, ["damaged_object"], {"tool_use": 5}
            )
        )

        # Poor condition should reduce success probability
        assert probability < 0.8  # Should be lower due to poor condition

    @pytest.mark.asyncio
    async def test_complex_interaction_chain(self, interaction_system):
        """Test complex interaction involving multiple steps."""
        # First interaction: examine object
        examine_result = await interaction_system.process_object_interaction(
            ObjectInteractionType.EXAMINE,
            "complex_object",
            None,
            None,
            {"player_state": {"skills": {}}},
        )

        # Second interaction: use object based on examination
        use_result = await interaction_system.process_object_interaction(
            ObjectInteractionType.USE,
            "complex_object",
            None,
            None,
            {"player_state": {"skills": {"tool_use": 5}}},
        )

        assert examine_result.success is True
        assert use_result.success is True
        assert examine_result.interaction_type != use_result.interaction_type

    @pytest.mark.asyncio
    async def test_error_handling_invalid_object(self, interaction_system):
        """Test error handling for invalid objects."""
        # Mock object that doesn't exist
        interaction_system.objects.object_exists = AsyncMock(return_value=False)

        result = await interaction_system.process_object_interaction(
            ObjectInteractionType.USE,
            "nonexistent_object",
            None,
            None,
            {"player_state": {"skills": {}}},
        )

        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_compatibility_matrix_caching(self, interaction_system):
        """Test that object compatibility is cached for performance."""
        # First call should calculate compatibility
        compatibility1 = await interaction_system._check_object_compatibility(
            "obj1", "obj2"
        )

        # Add to cache manually to test cache retrieval
        interaction_system._compatibility_matrix["obj1:obj2"] = {"compatibility": 0.8}

        # Second call should use cached value
        compatibility2 = await interaction_system._check_object_compatibility(
            "obj1", "obj2"
        )

        assert compatibility2 == 0.8  # Cached value
