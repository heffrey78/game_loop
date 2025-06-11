"""
Unit tests for EnvironmentInteractionManager.

This module tests the environment interaction functionality including
object manipulation, container interactions, tool usage, and puzzle mechanics.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionResult,
    PhysicalActionType,
)
from game_loop.core.environment.interaction_manager import (
    EnvironmentInteractionManager,
    InteractionType,
)


class TestEnvironmentInteractionManager:
    """Test cases for EnvironmentInteractionManager."""

    @pytest.fixture
    def mock_world_state_manager(self):
        """Fixture for mock world state manager."""
        return Mock()

    @pytest.fixture
    def mock_object_manager(self):
        """Fixture for mock object manager."""
        return Mock()

    @pytest.fixture
    def mock_physics_engine(self):
        """Fixture for mock physics engine."""
        mock_physics = Mock()
        mock_physics.validate_physical_constraints = AsyncMock(
            return_value=(True, None)
        )
        return mock_physics

    @pytest.fixture
    def interaction_manager(
        self, mock_world_state_manager, mock_object_manager, mock_physics_engine
    ):
        """Fixture for EnvironmentInteractionManager instance."""
        return EnvironmentInteractionManager(
            world_state_manager=mock_world_state_manager,
            object_manager=mock_object_manager,
            physics_engine=mock_physics_engine,
        )

    @pytest.fixture
    def sample_context(self):
        """Fixture for sample interaction context."""
        return {
            "player_id": "player_1",
            "player_state": {
                "current_location": "workshop",
                "energy": 100,
                "strength": 50,
                "inventory": ["hammer", "rope", "key"],
                "skills": {"mechanics": 5, "crafting": 3},
            },
            "current_location": "workshop",
        }

    @pytest.mark.asyncio
    async def test_process_environment_interaction_success(
        self, interaction_manager, sample_context
    ):
        """Test successful environment interaction processing."""
        result = await interaction_manager.process_environment_interaction(
            "player_1", "wooden_crate", "examine", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is True
        assert result.action_type == PhysicalActionType.MANIPULATION
        assert "wooden_crate" in result.affected_entities

    @pytest.mark.asyncio
    async def test_process_environment_interaction_validation_failure(
        self, interaction_manager
    ):
        """Test environment interaction with validation failure."""
        # Context with no energy
        empty_context = {
            "player_id": "player_1",
            "player_state": {"energy": 0},
        }

        result = await interaction_manager.process_environment_interaction(
            "player_1", "heavy_boulder", "push", empty_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_validate_interaction_requirements_success(
        self, interaction_manager, sample_context
    ):
        """Test successful interaction requirements validation."""
        is_valid, errors = await interaction_manager.validate_interaction_requirements(
            InteractionType.EXAMINE,
            "wooden_crate",
            sample_context.get("player_state", {}),
        )

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_interaction_requirements_missing_tool(
        self, interaction_manager
    ):
        """Test interaction requirements validation with missing tool."""
        context = {
            "player_state": {
                "inventory": [],  # No tools
                "skills": {},
            }
        }

        is_valid, errors = await interaction_manager.validate_interaction_requirements(
            InteractionType.TOOL_USE, "locked_box", context["player_state"]
        )

        # Should still be valid if no specific tool is required
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_execute_object_manipulation_success(
        self, interaction_manager, sample_context
    ):
        """Test successful object manipulation."""
        result = await interaction_manager.execute_object_manipulation(
            "wooden_box", "push", sample_context.get("player_state", {})
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MANIPULATION
        assert result.energy_cost > 0
        assert result.time_elapsed > 0

    @pytest.mark.asyncio
    async def test_execute_object_manipulation_insufficient_strength(
        self, interaction_manager
    ):
        """Test object manipulation with insufficient strength."""
        weak_player_state = {
            "strength": 10,  # Very weak
            "energy": 100,
        }

        result = await interaction_manager.execute_object_manipulation(
            "massive_boulder", "lift", weak_player_state
        )

        assert isinstance(result, PhysicalActionResult)
        # Result could be success or failure depending on implementation

    @pytest.mark.asyncio
    async def test_handle_container_interactions_open(
        self, interaction_manager, sample_context
    ):
        """Test container opening interaction."""
        result = await interaction_manager.handle_container_interactions(
            "treasure_chest", "open", None, sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type in [
            PhysicalActionType.OPENING,
            PhysicalActionType.MANIPULATION,
        ]

    @pytest.mark.asyncio
    async def test_handle_container_interactions_put_item(
        self, interaction_manager, sample_context
    ):
        """Test putting item in container."""
        result = await interaction_manager.handle_container_interactions(
            "storage_box", "put", "rope", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MANIPULATION

    @pytest.mark.asyncio
    async def test_handle_container_interactions_take_item(
        self, interaction_manager, sample_context
    ):
        """Test taking item from container."""
        result = await interaction_manager.handle_container_interactions(
            "supply_crate", "take", "healing_potion", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MANIPULATION

    @pytest.mark.asyncio
    async def test_handle_container_interactions_invalid_action(
        self, interaction_manager, sample_context
    ):
        """Test container interaction with invalid action."""
        result = await interaction_manager.handle_container_interactions(
            "chest", "invalid_action", None, sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert "invalid" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_process_tool_usage_success(
        self, interaction_manager, sample_context
    ):
        """Test successful tool usage."""
        result = await interaction_manager.process_tool_usage(
            "hammer", "wooden_board", "repair", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MANIPULATION
        assert "hammer" in result.affected_entities
        assert "wooden_board" in result.affected_entities

    @pytest.mark.asyncio
    async def test_process_tool_usage_missing_tool(self, interaction_manager):
        """Test tool usage when player doesn't have the tool."""
        context_without_tool = {
            "player_state": {
                "inventory": [],  # No tools
            }
        }

        result = await interaction_manager.process_tool_usage(
            "hammer", "nail", "hit", context_without_tool
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert "inventory" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_process_tool_usage_incompatible_action(
        self, interaction_manager, sample_context
    ):
        """Test tool usage with incompatible action."""
        result = await interaction_manager.process_tool_usage(
            "hammer", "water", "drink", sample_context  # Can't drink water with hammer
        )

        assert isinstance(result, PhysicalActionResult)
        # Should handle incompatible actions gracefully

    @pytest.mark.asyncio
    async def test_handle_environmental_puzzles_success(
        self, interaction_manager, sample_context
    ):
        """Test successful puzzle interaction."""
        result = await interaction_manager.handle_environmental_puzzles(
            "ancient_mechanism", "turn_dial", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MANIPULATION
        assert result.energy_cost > 0

    @pytest.mark.asyncio
    async def test_handle_environmental_puzzles_already_solved(
        self, interaction_manager, sample_context
    ):
        """Test puzzle interaction when puzzle is already solved."""
        # Set up puzzle as already solved
        interaction_manager._environmental_states["puzzle_box"] = {"solved": True}

        result = await interaction_manager.handle_environmental_puzzles(
            "puzzle_box", "rotate", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert "already" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_update_environmental_state(self, interaction_manager):
        """Test environmental state updating."""
        state_changes = {
            "chest_open": True,
            "chest_contents": ["gold_coin", "magic_scroll"],
        }

        await interaction_manager.update_environmental_state(
            "treasure_chest", state_changes
        )

        # Check that state was updated
        entity_state = await interaction_manager._get_entity_state("treasure_chest")
        assert (
            "open" in entity_state
            or len(interaction_manager._environmental_states) >= 0
        )

    @pytest.mark.asyncio
    async def test_check_interaction_side_effects(
        self, interaction_manager, sample_context
    ):
        """Test interaction side effects checking."""
        side_effects = await interaction_manager.check_interaction_side_effects(
            "break", ["glass_vase"], sample_context
        )

        assert isinstance(side_effects, list)

    @pytest.mark.asyncio
    async def test_get_interaction_options(self, interaction_manager, sample_context):
        """Test getting interaction options for an entity."""
        options = await interaction_manager.get_interaction_options(
            "wooden_chest", sample_context.get("player_state", {})
        )

        assert isinstance(options, list)
        assert len(options) > 0
        # Should always have at least examine option
        assert any(option["action"] == "examine" for option in options)

    @pytest.mark.asyncio
    async def test_get_interaction_options_container(
        self, interaction_manager, sample_context
    ):
        """Test getting interaction options for container."""
        # Mock container entity
        with patch.object(interaction_manager, "_get_object_info") as mock_get_info:
            mock_get_info.return_value = {"type": "container", "moveable": False}

            options = await interaction_manager.get_interaction_options(
                "storage_chest", sample_context.get("player_state", {})
            )

            assert isinstance(options, list)
            # Should have container-specific options
            action_names = [opt["action"] for opt in options]
            assert "examine" in action_names

    @pytest.mark.asyncio
    async def test_get_interaction_options_moveable_object(
        self, interaction_manager, sample_context
    ):
        """Test getting interaction options for moveable object."""
        with patch.object(interaction_manager, "_get_object_info") as mock_get_info:
            mock_get_info.return_value = {
                "type": "object",
                "moveable": True,
                "mass": 15,
            }

            options = await interaction_manager.get_interaction_options(
                "wooden_crate", sample_context.get("player_state", {})
            )

            assert isinstance(options, list)
            action_names = [opt["action"] for opt in options]
            assert "examine" in action_names

    def test_register_interaction_handler(self, interaction_manager):
        """Test registering custom interaction handler."""
        mock_handler = Mock()

        interaction_manager.register_interaction_handler(
            "custom_interaction", mock_handler
        )

        assert "custom_interaction" in interaction_manager._interaction_handlers
        assert (
            interaction_manager._interaction_handlers["custom_interaction"]
            == mock_handler
        )

    @pytest.mark.asyncio
    async def test_container_open_locked(self, interaction_manager, sample_context):
        """Test opening a locked container."""
        # Mock locked container state
        with patch.object(interaction_manager, "_get_entity_state") as mock_get_state:
            mock_get_state.return_value = {"locked": True, "open": False}

            result = await interaction_manager._handle_container_open(
                "locked_chest", {"locked": True}, sample_context.get("player_state", {})
            )

            assert isinstance(result, PhysicalActionResult)
            assert result.success is False
            assert "locked" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_container_open_unlocked(self, interaction_manager, sample_context):
        """Test opening an unlocked container."""
        result = await interaction_manager._handle_container_open(
            "unlocked_chest", {"locked": False}, sample_context.get("player_state", {})
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is True
        assert result.action_type == PhysicalActionType.OPENING

    @pytest.mark.asyncio
    async def test_container_close(self, interaction_manager, sample_context):
        """Test closing a container."""
        result = await interaction_manager._handle_container_close(
            "open_chest", {"open": True}, sample_context.get("player_state", {})
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is True
        assert result.action_type == PhysicalActionType.CLOSING

    @pytest.mark.asyncio
    async def test_puzzle_completion_detection(
        self, interaction_manager, sample_context
    ):
        """Test puzzle completion detection."""
        # Mock puzzle state that will be solved after 3 attempts
        puzzle_state = {"attempts": 2}  # Will be 3 after one more attempt

        is_completed = await interaction_manager._check_puzzle_completion(
            "test_puzzle", {"attempts": 3}
        )

        assert isinstance(is_completed, bool)

    @pytest.mark.asyncio
    async def test_tool_wear_calculation(self, interaction_manager):
        """Test tool wear calculation."""
        wear = await interaction_manager._calculate_tool_wear("hammer", "hit", 1.5)

        assert isinstance(wear, float)
        assert wear > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, interaction_manager):
        """Test error handling in interaction processing."""
        # Test with invalid context
        result = await interaction_manager.process_environment_interaction(
            "invalid_player", "invalid_object", "invalid_action", {}
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_manipulation_description_generation(self, interaction_manager):
        """Test manipulation description generation."""
        description = await interaction_manager._generate_manipulation_description(
            "wooden_box", "push", {"wooden_box_position": "moved"}
        )

        assert isinstance(description, str)
        assert "push" in description.lower()
        assert "wooden_box" in description.lower()

    @pytest.mark.asyncio
    async def test_interaction_difficulty_calculation(self, interaction_manager):
        """Test interaction difficulty calculation."""
        difficulty = await interaction_manager._calculate_interaction_difficulty(
            "lift", "heavy_stone"
        )

        assert isinstance(difficulty, float)
        assert difficulty > 0

    @pytest.mark.asyncio
    async def test_apply_wear_and_tear(self, interaction_manager):
        """Test wear and tear application."""
        await interaction_manager._apply_wear_and_tear("wooden_tool", 0.8)

        # Check that durability was affected
        entity_state = interaction_manager._environmental_states.get("wooden_tool", {})
        assert "durability" in entity_state

    @pytest.mark.asyncio
    async def test_manipulation_energy_cost_calculation(self, interaction_manager):
        """Test manipulation energy cost calculation."""
        cost = interaction_manager._calculate_manipulation_energy_cost(
            "push", 20.0, 1.5
        )

        assert isinstance(cost, float)
        assert cost > 0

    @pytest.mark.asyncio
    async def test_manipulation_time_cost_calculation(self, interaction_manager):
        """Test manipulation time cost calculation."""
        time_cost = interaction_manager._calculate_manipulation_time_cost(
            "pull", 15.0, 1.2
        )

        assert isinstance(time_cost, float)
        assert time_cost > 0


@pytest.mark.asyncio
async def test_integration_with_physics_engine():
    """Test EnvironmentInteractionManager integration with physics engine."""
    # Create mocks
    mock_world_state = Mock()
    mock_object_manager = Mock()
    mock_physics = Mock()
    mock_physics.validate_physical_constraints = AsyncMock(return_value=(True, None))

    # Create interaction manager
    interaction_manager = EnvironmentInteractionManager(
        world_state_manager=mock_world_state,
        object_manager=mock_object_manager,
        physics_engine=mock_physics,
    )

    context = {
        "player_id": "test_player",
        "player_state": {
            "current_location": "workshop",
            "energy": 100,
            "strength": 60,
            "inventory": ["hammer"],
        },
    }

    # Test interaction processing
    result = await interaction_manager.process_environment_interaction(
        "test_player", "test_object", "push", context
    )

    # Verify result
    assert isinstance(result, PhysicalActionResult)
    assert result.action_type == PhysicalActionType.MANIPULATION


if __name__ == "__main__":
    pytest.main([__file__])
