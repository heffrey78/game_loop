"""
Unit tests for PhysicalActionProcessor.

This module tests the core physical action processing functionality including
action validation, execution, and result generation.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from game_loop.core.actions.types import ActionClassification, ActionType
from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionProcessor,
    PhysicalActionResult,
    PhysicalActionType,
)


class TestPhysicalActionProcessor:
    """Test cases for PhysicalActionProcessor."""

    @pytest.fixture
    def mock_console(self):
        """Fixture for mock console."""
        return Mock(spec=Console)

    @pytest.fixture
    def mock_game_state_manager(self):
        """Fixture for mock game state manager."""
        return Mock()

    @pytest.fixture
    def mock_search_service(self):
        """Fixture for mock search service."""
        return Mock()

    @pytest.fixture
    def mock_physics_engine(self):
        """Fixture for mock physics engine."""
        mock_physics = Mock()
        mock_physics.validate_physical_constraints = AsyncMock(
            return_value=(True, None)
        )
        mock_physics.get_constraint_violations = AsyncMock(return_value=[])
        return mock_physics

    @pytest.fixture
    def processor(
        self,
        mock_console,
        mock_game_state_manager,
        mock_search_service,
        mock_physics_engine,
    ):
        """Fixture for PhysicalActionProcessor instance."""
        return PhysicalActionProcessor(
            console=mock_console,
            game_state_manager=mock_game_state_manager,
            search_service=mock_search_service,
            physics_engine=mock_physics_engine,
        )

    @pytest.fixture
    def sample_action_classification(self):
        """Fixture for sample action classification."""
        return ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="rock",
            intent="push the rock",
            raw_input="push the rock",
            parameters={"force": "moderate"},
        )

    @pytest.fixture
    def sample_context(self):
        """Fixture for sample game context."""
        return {
            "player_id": "player_1",
            "player_state": {
                "current_location": "forest_clearing",
                "energy": 100,
                "strength": 50,
                "inventory": ["torch", "rope"],
            },
            "current_location": "forest_clearing",
        }

    @pytest.mark.asyncio
    async def test_process_physical_action_success(
        self, processor, sample_action_classification, sample_context
    ):
        """Test successful physical action processing."""
        result = await processor.process_physical_action(
            sample_action_classification, sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is True
        assert result.action_type in PhysicalActionType
        assert result.energy_cost > 0
        assert result.time_elapsed > 0
        assert isinstance(result.affected_entities, list)
        assert isinstance(result.state_changes, dict)

    @pytest.mark.asyncio
    async def test_process_physical_action_insufficient_energy(
        self, processor, sample_action_classification
    ):
        """Test physical action processing with insufficient energy."""
        low_energy_context = {
            "player_id": "player_1",
            "player_state": {
                "current_location": "forest_clearing",
                "energy": 5,  # Very low energy
                "strength": 50,
            },
        }

        result = await processor.process_physical_action(
            sample_action_classification, low_energy_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert "energy" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_validate_action_feasibility_valid(self, processor, sample_context):
        """Test action feasibility validation for valid action."""
        is_feasible, error_msg = await processor.validate_action_feasibility(
            PhysicalActionType.PUSHING, ["rock"], sample_context
        )

        assert is_feasible is True
        assert error_msg is None

    @pytest.mark.asyncio
    async def test_validate_action_feasibility_no_player(self, processor):
        """Test action feasibility validation without player context."""
        empty_context = {}

        is_feasible, error_msg = await processor.validate_action_feasibility(
            PhysicalActionType.PUSHING, ["rock"], empty_context
        )

        assert is_feasible is False
        assert "player" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_calculate_action_requirements(self, processor):
        """Test action requirements calculation."""
        requirements = await processor.calculate_action_requirements(
            PhysicalActionType.CLIMBING, ["cliff_wall"]
        )

        assert isinstance(requirements, dict)
        assert "energy" in requirements
        assert "time" in requirements
        assert "difficulty" in requirements
        assert requirements["energy"] > 0
        assert requirements["time"] > 0

    @pytest.mark.asyncio
    async def test_execute_movement_action(self, processor, sample_context):
        """Test movement action execution."""
        result = await processor.execute_movement_action("north", None, sample_context)

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MOVEMENT
        assert "north" in result.description.lower()

    @pytest.mark.asyncio
    async def test_execute_movement_action_no_player(self, processor):
        """Test movement action execution without player."""
        empty_context = {}

        result = await processor.execute_movement_action("north", None, empty_context)

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert "player" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_manipulation_action(self, processor, sample_context):
        """Test object manipulation action execution."""
        result = await processor.execute_manipulation_action(
            "box", "push", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type in [
            PhysicalActionType.PUSHING,
            PhysicalActionType.MANIPULATION,
        ]
        assert "box" in result.description.lower()
        assert "push" in result.description.lower()

    @pytest.mark.asyncio
    async def test_execute_environmental_action(self, processor, sample_context):
        """Test environmental action execution."""
        result = await processor.execute_environmental_action(
            "tree", PhysicalActionType.CLIMBING, sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.CLIMBING
        assert result.energy_cost > 0

    @pytest.mark.asyncio
    async def test_apply_physics_constraints(
        self, processor, mock_physics_engine, sample_context
    ):
        """Test physics constraints application."""
        mock_physics_engine.get_constraint_violations = AsyncMock(
            return_value=[
                {"rule": "mass_limit", "violation": "Too heavy", "severity": "high"}
            ]
        )

        constraints = await processor.apply_physics_constraints(
            PhysicalActionType.PUSHING, ["heavy_boulder"], sample_context
        )

        assert isinstance(constraints, list)
        assert len(constraints) > 0
        assert constraints[0]["rule"] == "mass_limit"

    @pytest.mark.asyncio
    async def test_update_world_state(
        self, processor, mock_game_state_manager, sample_context
    ):
        """Test world state update."""
        action_result = PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.PUSHING,
            affected_entities=["rock"],
            state_changes={"rock_position": "moved"},
            energy_cost=15.0,
            time_elapsed=5.0,
            side_effects=[],
            description="Rock pushed successfully.",
        )

        # Should not raise an exception
        await processor.update_world_state(action_result, sample_context)

    @pytest.mark.asyncio
    async def test_calculate_side_effects(self, processor, sample_context):
        """Test side effects calculation."""
        side_effects = await processor.calculate_side_effects(
            PhysicalActionType.BREAKING, ["wooden_crate"], sample_context
        )

        assert isinstance(side_effects, list)
        # Breaking actions should have debris side effect
        assert any("debris" in effect.lower() for effect in side_effects)

    def test_register_action_handler(self, processor):
        """Test action handler registration."""
        mock_handler = Mock()

        processor.register_action_handler(PhysicalActionType.JUMPING, mock_handler)

        assert PhysicalActionType.JUMPING in processor._action_handlers
        assert processor._action_handlers[PhysicalActionType.JUMPING] == mock_handler

    def test_register_constraint_validator(self, processor):
        """Test constraint validator registration."""
        mock_validator = Mock()

        processor.register_constraint_validator(mock_validator)

        assert mock_validator in processor._constraint_validators

    @pytest.mark.asyncio
    async def test_determine_physical_action_type_movement(
        self, processor, sample_context
    ):
        """Test physical action type determination for movement."""
        movement_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="north",
            intent="go north",
        )

        action_type = await processor._determine_physical_action_type(
            movement_classification, sample_context
        )

        assert action_type == PhysicalActionType.MOVEMENT

    @pytest.mark.asyncio
    async def test_determine_physical_action_type_climbing(
        self, processor, sample_context
    ):
        """Test physical action type determination for climbing."""
        climbing_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="tree",
            intent="climb the tree",
        )

        action_type = await processor._determine_physical_action_type(
            climbing_classification, sample_context
        )

        assert action_type == PhysicalActionType.CLIMBING

    def test_get_manipulation_type_push(self, processor):
        """Test manipulation type determination for pushing."""
        manipulation_type = processor._get_manipulation_type("push the boulder")

        assert manipulation_type == PhysicalActionType.PUSHING

    def test_get_manipulation_type_pull(self, processor):
        """Test manipulation type determination for pulling."""
        manipulation_type = processor._get_manipulation_type("pull the rope")

        assert manipulation_type == PhysicalActionType.PULLING

    def test_normalize_direction(self, processor):
        """Test direction normalization."""
        assert processor._normalize_direction("N") == "north"
        assert processor._normalize_direction("NORTH") == "north"
        assert processor._normalize_direction("South") == "south"
        assert processor._normalize_direction("unknown") == "unknown"

    @pytest.mark.asyncio
    async def test_execute_action_by_type_movement(
        self, processor, sample_action_classification, sample_context
    ):
        """Test action execution by type for movement."""
        movement_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target="west",
            intent="move west",
        )

        requirements = {"energy": 5.0, "time": 3.0, "difficulty": 0.2}

        result = await processor._execute_action_by_type(
            PhysicalActionType.MOVEMENT,
            movement_classification,
            sample_context,
            requirements,
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MOVEMENT

    @pytest.mark.asyncio
    async def test_update_action_metrics(self, processor):
        """Test action metrics update."""
        action_result = PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.JUMPING,
            affected_entities=[],
            state_changes={},
            energy_cost=10.0,
            time_elapsed=3.0,
            side_effects=[],
            description="Jump completed.",
        )

        await processor._update_action_metrics(
            PhysicalActionType.JUMPING, action_result
        )

        metrics = processor.get_action_metrics()
        assert "jumping" in metrics
        assert metrics["jumping"]["total_attempts"] == 1
        assert metrics["jumping"]["successful_attempts"] == 1

    @pytest.mark.asyncio
    async def test_calculate_energy_cost(self, processor):
        """Test energy cost calculation."""
        cost = await processor._calculate_energy_cost(PhysicalActionType.CLIMBING, 0.8)

        assert isinstance(cost, float)
        assert cost > 0
        # Climbing should be more expensive than basic movement
        movement_cost = await processor._calculate_energy_cost(
            PhysicalActionType.MOVEMENT, 0.2
        )
        assert cost > movement_cost

    @pytest.mark.asyncio
    async def test_validate_entity_accessibility(self, processor, sample_context):
        """Test entity accessibility validation."""
        is_accessible = await processor._validate_entity_accessibility(
            "nearby_rock", sample_context
        )

        # In the simplified implementation, this should return True
        assert is_accessible is True

    @pytest.mark.asyncio
    async def test_physics_engine_integration(
        self,
        processor,
        mock_physics_engine,
        sample_action_classification,
        sample_context,
    ):
        """Test integration with physics engine."""
        # Configure physics engine to reject action
        mock_physics_engine.validate_physical_constraints = AsyncMock(
            return_value=(False, "Physics violation")
        )

        result = await processor.process_physical_action(
            sample_action_classification, sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert "physics violation" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in action processing."""
        # Test with invalid classification
        invalid_classification = ActionClassification(
            action_type=ActionType.PHYSICAL,
            confidence=0.9,
            target=None,
            intent=None,
        )

        # This should not raise an exception but return an error result
        result = await processor.process_physical_action(invalid_classification, {})

        assert isinstance(result, PhysicalActionResult)
        assert result.success is False
        assert result.error_message is not None


class TestPhysicalActionResult:
    """Test cases for PhysicalActionResult."""

    def test_action_result_creation(self):
        """Test PhysicalActionResult creation."""
        result = PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.PUSHING,
            affected_entities=["rock", "tree"],
            state_changes={"rock_position": "new_location"},
            energy_cost=15.5,
            time_elapsed=3.2,
            side_effects=["dust_cloud"],
            description="Successfully pushed the rock against the tree.",
        )

        assert result.success is True
        assert result.action_type == PhysicalActionType.PUSHING
        assert len(result.affected_entities) == 2
        assert "rock_position" in result.state_changes
        assert result.energy_cost == 15.5
        assert result.time_elapsed == 3.2
        assert "dust_cloud" in result.side_effects

    def test_action_result_to_dict(self):
        """Test PhysicalActionResult to dictionary conversion."""
        result = PhysicalActionResult(
            success=False,
            action_type=PhysicalActionType.CLIMBING,
            affected_entities=["cliff"],
            state_changes={},
            energy_cost=25.0,
            time_elapsed=8.0,
            side_effects=[],
            description="Failed to climb the cliff.",
            error_message="Insufficient grip strength",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is False
        assert result_dict["action_type"] == "climbing"
        assert result_dict["affected_entities"] == ["cliff"]
        assert result_dict["energy_cost"] == 25.0
        assert result_dict["error_message"] == "Insufficient grip strength"


class TestPhysicalActionType:
    """Test cases for PhysicalActionType enum."""

    def test_physical_action_type_values(self):
        """Test PhysicalActionType enum values."""
        assert PhysicalActionType.MOVEMENT.value == "movement"
        assert PhysicalActionType.MANIPULATION.value == "manipulation"
        assert PhysicalActionType.CLIMBING.value == "climbing"
        assert PhysicalActionType.JUMPING.value == "jumping"
        assert PhysicalActionType.PUSHING.value == "pushing"
        assert PhysicalActionType.PULLING.value == "pulling"
        assert PhysicalActionType.LIFTING.value == "lifting"
        assert PhysicalActionType.OPENING.value == "opening"
        assert PhysicalActionType.CLOSING.value == "closing"
        assert PhysicalActionType.BREAKING.value == "breaking"
        assert PhysicalActionType.BUILDING.value == "building"

    def test_physical_action_type_from_string(self):
        """Test creating PhysicalActionType from string."""
        movement_type = PhysicalActionType("movement")
        assert movement_type == PhysicalActionType.MOVEMENT

        pushing_type = PhysicalActionType("pushing")
        assert pushing_type == PhysicalActionType.PUSHING


@pytest.mark.asyncio
async def test_integration_with_mock_dependencies():
    """Test PhysicalActionProcessor integration with all mock dependencies."""
    # Create all mocks
    mock_console = Mock(spec=Console)
    mock_game_state = Mock()
    mock_search = Mock()
    mock_physics = Mock()
    mock_physics.validate_physical_constraints = AsyncMock(return_value=(True, None))
    mock_physics.get_constraint_violations = AsyncMock(return_value=[])

    # Create processor
    processor = PhysicalActionProcessor(
        console=mock_console,
        game_state_manager=mock_game_state,
        search_service=mock_search,
        physics_engine=mock_physics,
    )

    # Create test action
    action = ActionClassification(
        action_type=ActionType.PHYSICAL,
        confidence=0.95,
        target="heavy_box",
        intent="push the heavy box to the corner",
        parameters={"direction": "northeast", "force": "strong"},
    )

    context = {
        "player_id": "test_player",
        "player_state": {
            "current_location": "warehouse",
            "energy": 80,
            "strength": 60,
            "skills": {"athletics": 5},
        },
    }

    # Execute action
    result = await processor.process_physical_action(action, context)

    # Verify result
    assert isinstance(result, PhysicalActionResult)
    assert result.action_type in PhysicalActionType

    # Verify mocks were called appropriately
    mock_physics.validate_physical_constraints.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
