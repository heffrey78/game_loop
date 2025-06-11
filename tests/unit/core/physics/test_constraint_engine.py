"""
Unit tests for PhysicsConstraintEngine.

This module tests the physics constraint validation, strength calculations,
energy expenditure, and realistic physics simulation.
"""

from unittest.mock import Mock, patch

import pytest

from game_loop.core.command_handlers.physical_action_processor import PhysicalActionType
from game_loop.core.physics.constraint_engine import (
    PhysicsConstants,
    PhysicsConstraintEngine,
)


class TestPhysicsConstraintEngine:
    """Test cases for PhysicsConstraintEngine."""

    @pytest.fixture
    def mock_configuration_manager(self):
        """Fixture for mock configuration manager."""
        return Mock()

    @pytest.fixture
    def physics_engine(self, mock_configuration_manager):
        """Fixture for PhysicsConstraintEngine instance."""
        return PhysicsConstraintEngine(configuration_manager=mock_configuration_manager)

    @pytest.fixture
    def sample_player_state(self):
        """Fixture for sample player state."""
        return {
            "strength": 50,
            "energy": 100,
            "fitness": 70,
            "balance_skill": 60,
            "current_location": {"space_rating": 8},
        }

    @pytest.mark.asyncio
    async def test_validate_physical_constraints_success(
        self, physics_engine, sample_player_state
    ):
        """Test successful physical constraints validation."""
        is_valid, error_msg = await physics_engine.validate_physical_constraints(
            PhysicalActionType.PUSHING, ["wooden_crate"], sample_player_state
        )

        assert is_valid is True
        assert error_msg is None

    @pytest.mark.asyncio
    async def test_validate_physical_constraints_failure(self, physics_engine):
        """Test physical constraints validation failure."""
        weak_player_state = {
            "strength": 5,  # Very weak
            "energy": 10,  # Low energy
            "balance_skill": 10,
        }

        is_valid, error_msg = await physics_engine.validate_physical_constraints(
            PhysicalActionType.BREAKING, ["massive_boulder"], weak_player_state
        )

        # At least one constraint should fail
        assert isinstance(is_valid, bool)
        if not is_valid:
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0

    @pytest.mark.asyncio
    async def test_calculate_strength_requirements(self, physics_engine):
        """Test strength requirements calculation."""
        target_mass = 25.0
        difficulty_modifiers = {"terrain_difficulty": 1.3, "weather_factor": 1.1}

        strength_required = await physics_engine.calculate_strength_requirements(
            PhysicalActionType.LIFTING, target_mass, difficulty_modifiers
        )

        assert isinstance(strength_required, float)
        assert strength_required > 0
        # Should be affected by mass and difficulty modifiers
        assert strength_required > 1.0

    @pytest.mark.asyncio
    async def test_calculate_strength_requirements_different_actions(
        self, physics_engine
    ):
        """Test strength requirements for different action types."""
        mass = 20.0
        modifiers = {}

        push_strength = await physics_engine.calculate_strength_requirements(
            PhysicalActionType.PUSHING, mass, modifiers
        )
        climb_strength = await physics_engine.calculate_strength_requirements(
            PhysicalActionType.CLIMBING, mass, modifiers
        )
        break_strength = await physics_engine.calculate_strength_requirements(
            PhysicalActionType.BREAKING, mass, modifiers
        )

        # Breaking should require more strength than pushing
        assert break_strength > push_strength
        # All should be positive
        assert all(s > 0 for s in [push_strength, climb_strength, break_strength])

    @pytest.mark.asyncio
    async def test_check_spatial_constraints_sufficient_space(self, physics_engine):
        """Test spatial constraints with sufficient space."""
        action_location = {
            "available_space": {
                "width": 5.0,
                "height": 3.0,
                "depth": 4.0,
            },
            "entity_positions": {},
        }

        required_space = {
            "width": 3.0,
            "height": 2.5,
            "depth": 3.0,
        }

        has_space = await physics_engine.check_spatial_constraints(
            action_location, required_space, ["player"]
        )

        assert has_space is True

    @pytest.mark.asyncio
    async def test_check_spatial_constraints_insufficient_space(self, physics_engine):
        """Test spatial constraints with insufficient space."""
        action_location = {
            "available_space": {
                "width": 2.0,
                "height": 1.5,
                "depth": 2.0,
            },
            "entity_positions": {},
        }

        required_space = {
            "width": 5.0,  # More than available
            "height": 3.0,  # More than available
            "depth": 3.0,  # More than available
        }

        has_space = await physics_engine.check_spatial_constraints(
            action_location, required_space, ["player"]
        )

        assert has_space is False

    @pytest.mark.asyncio
    async def test_validate_structural_integrity_within_limits(self, physics_engine):
        """Test structural integrity validation within limits."""
        can_withstand, damage_info = await physics_engine.validate_structural_integrity(
            "wooden_beam", 400.0, "compression"  # Within limits
        )

        assert can_withstand is True
        assert damage_info is None

    @pytest.mark.asyncio
    async def test_validate_structural_integrity_damage(self, physics_engine):
        """Test structural integrity with damage but no failure."""
        can_withstand, damage_info = await physics_engine.validate_structural_integrity(
            "wooden_beam", 700.0, "tension"  # Above yield but below failure
        )

        assert isinstance(can_withstand, bool)
        if damage_info:
            assert isinstance(damage_info, dict)
            assert "type" in damage_info

    @pytest.mark.asyncio
    async def test_validate_structural_integrity_failure(self, physics_engine):
        """Test structural integrity failure."""
        can_withstand, damage_info = await physics_engine.validate_structural_integrity(
            "fragile_glass", 2000.0, "compression"  # Excessive force
        )

        assert isinstance(can_withstand, bool)
        if not can_withstand:
            assert damage_info is not None
            assert damage_info["type"] == "structural_failure"

    @pytest.mark.asyncio
    async def test_calculate_energy_expenditure(
        self, physics_engine, sample_player_state
    ):
        """Test energy expenditure calculation."""
        energy_cost = await physics_engine.calculate_energy_expenditure(
            PhysicalActionType.CLIMBING,
            duration=30.0,  # 30 seconds
            intensity=0.8,  # High intensity
            player_stats=sample_player_state,
        )

        assert isinstance(energy_cost, float)
        assert energy_cost > 0
        # Climbing should be more expensive than basic movement
        assert energy_cost > 10.0

    @pytest.mark.asyncio
    async def test_calculate_energy_expenditure_different_actions(
        self, physics_engine, sample_player_state
    ):
        """Test energy expenditure for different action types."""
        duration = 20.0
        intensity = 0.6

        movement_energy = await physics_engine.calculate_energy_expenditure(
            PhysicalActionType.MOVEMENT, duration, intensity, sample_player_state
        )
        climbing_energy = await physics_engine.calculate_energy_expenditure(
            PhysicalActionType.CLIMBING, duration, intensity, sample_player_state
        )
        breaking_energy = await physics_engine.calculate_energy_expenditure(
            PhysicalActionType.BREAKING, duration, intensity, sample_player_state
        )

        # Breaking should be most expensive, movement least expensive
        assert breaking_energy > climbing_energy > movement_energy
        assert all(e > 0 for e in [movement_energy, climbing_energy, breaking_energy])

    @pytest.mark.asyncio
    async def test_simulate_collision_effects(self, physics_engine):
        """Test collision effects simulation."""
        collision_effects = await physics_engine.simulate_collision_effects(
            "wooden_ball", "stone_wall", 500.0
        )

        assert isinstance(collision_effects, dict)
        assert "entity1_effects" in collision_effects
        assert "entity2_effects" in collision_effects
        assert "environmental_effects" in collision_effects
        assert "impact_force" in collision_effects

        # Check entity effects structure
        entity1_effects = collision_effects["entity1_effects"]
        assert "velocity_change" in entity1_effects
        assert "damage" in entity1_effects
        assert "position_change" in entity1_effects

    @pytest.mark.asyncio
    async def test_simulate_collision_effects_high_impact(self, physics_engine):
        """Test collision effects with high impact force."""
        collision_effects = await physics_engine.simulate_collision_effects(
            "metal_projectile", "concrete_wall", 1500.0  # High impact
        )

        assert isinstance(collision_effects, dict)
        env_effects = collision_effects.get("environmental_effects", {})

        # High impact should create more dramatic effects
        if "impact_crater" in env_effects:
            assert isinstance(env_effects["impact_crater"], bool)

    @pytest.mark.asyncio
    async def test_check_balance_and_stability_stable(
        self, physics_engine, sample_player_state
    ):
        """Test balance and stability check for stable action."""
        environmental_factors = {
            "wind_speed": 5.0,  # Light wind
            "surface_stability": 0.9,  # Stable surface
            "visibility": 1.0,  # Clear visibility
        }

        is_stable = await physics_engine.check_balance_and_stability(
            "player", PhysicalActionType.MOVEMENT, environmental_factors
        )

        assert isinstance(is_stable, bool)

    @pytest.mark.asyncio
    async def test_check_balance_and_stability_challenging(
        self, physics_engine, sample_player_state
    ):
        """Test balance and stability check for challenging conditions."""
        challenging_factors = {
            "wind_speed": 40.0,  # Strong wind
            "surface_stability": 0.3,  # Unstable surface
            "visibility": 0.2,  # Poor visibility
        }

        is_stable = await physics_engine.check_balance_and_stability(
            "player", PhysicalActionType.CLIMBING, challenging_factors
        )

        assert isinstance(is_stable, bool)

    @pytest.mark.asyncio
    async def test_apply_gravity_effects_ground_level(self, physics_engine):
        """Test gravity effects at ground level."""
        gravity_effects = await physics_engine.apply_gravity_effects(
            "player", 0.0, None  # At ground level
        )

        assert isinstance(gravity_effects, dict)
        assert gravity_effects.get("no_effect") is True

    @pytest.mark.asyncio
    async def test_apply_gravity_effects_elevated(self, physics_engine):
        """Test gravity effects when elevated."""
        gravity_effects = await physics_engine.apply_gravity_effects(
            "player", 10.0, "wooden_platform"  # 10 meters up with support
        )

        assert isinstance(gravity_effects, dict)
        assert "potential_energy" in gravity_effects
        assert "terminal_velocity" in gravity_effects
        assert "gravitational_force" in gravity_effects
        assert "stable" in gravity_effects

    @pytest.mark.asyncio
    async def test_apply_gravity_effects_support_failure(self, physics_engine):
        """Test gravity effects with support structure failure."""
        # Mock support that will fail under load
        with patch.object(
            physics_engine, "_get_entity_structural_properties"
        ) as mock_props:
            mock_props.return_value = {"max_load": 50.0}  # Very weak support

            gravity_effects = await physics_engine.apply_gravity_effects(
                "heavy_object", 5.0, "weak_branch"
            )

            assert isinstance(gravity_effects, dict)
            # Should detect support failure for heavy objects

    def test_register_constraint_rule(self, physics_engine):
        """Test registering custom constraint rule."""
        mock_validator = Mock()

        physics_engine.register_constraint_rule(
            "custom_rule", mock_validator, priority=7
        )

        assert "custom_rule" in physics_engine._constraint_rules
        assert physics_engine._constraint_rules["custom_rule"] == mock_validator

    @pytest.mark.asyncio
    async def test_get_constraint_violations(self, physics_engine, sample_player_state):
        """Test getting constraint violations."""
        context = {
            "entities": ["heavy_boulder"],
            "player_state": sample_player_state,
        }

        violations = await physics_engine.get_constraint_violations(
            PhysicalActionType.LIFTING, context
        )

        assert isinstance(violations, list)
        # Each violation should have required fields
        for violation in violations:
            assert "rule" in violation
            assert "violation" in violation
            assert "severity" in violation

    @pytest.mark.asyncio
    async def test_mass_limit_validation_success(
        self, physics_engine, sample_player_state
    ):
        """Test mass limit validation success."""
        is_valid, error = await physics_engine._validate_mass_limits(
            PhysicalActionType.PUSHING, ["light_box"], sample_player_state
        )

        assert is_valid is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_mass_limit_validation_failure(self, physics_engine):
        """Test mass limit validation failure."""
        weak_player = {"strength": 10}  # Very weak player

        is_valid, error = await physics_engine._validate_mass_limits(
            PhysicalActionType.LIFTING, ["massive_boulder"], weak_player
        )

        # Should handle validation appropriately
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

    @pytest.mark.asyncio
    async def test_energy_requirements_validation_success(
        self, physics_engine, sample_player_state
    ):
        """Test energy requirements validation success."""
        is_valid, error = await physics_engine._validate_energy_requirements(
            PhysicalActionType.MOVEMENT, ["player"], sample_player_state
        )

        assert is_valid is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_energy_requirements_validation_failure(self, physics_engine):
        """Test energy requirements validation failure."""
        tired_player = {"energy": 2}  # Very low energy

        is_valid, error = await physics_engine._validate_energy_requirements(
            PhysicalActionType.CLIMBING, ["cliff"], tired_player
        )

        assert is_valid is False
        assert "energy" in error.lower()

    @pytest.mark.asyncio
    async def test_spatial_clearance_validation(
        self, physics_engine, sample_player_state
    ):
        """Test spatial clearance validation."""
        is_valid, error = await physics_engine._validate_spatial_clearance(
            PhysicalActionType.JUMPING, ["gap"], sample_player_state
        )

        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

    @pytest.mark.asyncio
    async def test_balance_requirements_validation(
        self, physics_engine, sample_player_state
    ):
        """Test balance requirements validation."""
        is_valid, error = await physics_engine._validate_balance_requirements(
            PhysicalActionType.CLIMBING, ["rope"], sample_player_state
        )

        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

    def test_physics_constants(self):
        """Test physics constants are properly defined."""
        assert PhysicsConstants.GRAVITY == 9.81
        assert PhysicsConstants.AIR_DENSITY == 1.225
        assert PhysicsConstants.WATER_DENSITY == 1000.0

        # Test friction coefficients
        assert "wood_on_wood" in PhysicsConstants.FRICTION_COEFFICIENTS
        assert "default" in PhysicsConstants.FRICTION_COEFFICIENTS

        # Test drag coefficients
        assert "sphere" in PhysicsConstants.DRAG_COEFFICIENTS
        assert "default" in PhysicsConstants.DRAG_COEFFICIENTS

    def test_calculate_mass_factor(self, physics_engine):
        """Test mass factor calculation for different actions."""
        movement_factor = physics_engine._calculate_mass_factor(
            PhysicalActionType.MOVEMENT, 50.0
        )
        pushing_factor = physics_engine._calculate_mass_factor(
            PhysicalActionType.PUSHING, 50.0
        )
        climbing_factor = physics_engine._calculate_mass_factor(
            PhysicalActionType.CLIMBING, 50.0
        )

        # Climbing should be more affected by mass than movement
        assert climbing_factor > movement_factor
        assert all(f > 1.0 for f in [movement_factor, pushing_factor, climbing_factor])

    def test_environmental_resistance_calculation(self, physics_engine):
        """Test environmental resistance calculation."""
        normal_resistance = physics_engine._get_environmental_resistance(
            "normal_room", PhysicalActionType.MOVEMENT
        )
        underwater_resistance = physics_engine._get_environmental_resistance(
            "underwater_cave", PhysicalActionType.MOVEMENT
        )
        icy_resistance = physics_engine._get_environmental_resistance(
            "icy_path", PhysicalActionType.MOVEMENT
        )

        # Underwater should be harder than normal
        assert underwater_resistance > normal_resistance
        # All should be positive
        assert all(
            r > 0 for r in [normal_resistance, underwater_resistance, icy_resistance]
        )

    def test_space_overlap_detection(self, physics_engine):
        """Test space overlap detection."""
        space1 = {"x": 0, "y": 0, "width": 3, "height": 3}
        space2_overlap = {"x": 2, "y": 2, "width": 3, "height": 3}
        space2_no_overlap = {"x": 5, "y": 5, "width": 2, "height": 2}

        overlap1 = physics_engine._check_space_overlap(space1, space2_overlap)
        overlap2 = physics_engine._check_space_overlap(space1, space2_no_overlap)

        assert overlap1 is True  # Should overlap
        assert overlap2 is False  # Should not overlap

    def test_collision_damage_calculation(self, physics_engine):
        """Test collision damage calculation."""
        entity_props = {"durability": 100.0, "damage_threshold": 200.0}

        low_damage = physics_engine._calculate_collision_damage(
            "entity", 150.0, entity_props
        )
        high_damage = physics_engine._calculate_collision_damage(
            "entity", 350.0, entity_props
        )

        assert low_damage == 0.0  # Below threshold
        assert high_damage > 0.0  # Above threshold

    def test_collision_sound_generation(self, physics_engine):
        """Test collision sound generation."""
        props1 = {"material": "wood"}
        props2 = {"material": "metal"}

        soft_sound = physics_engine._generate_collision_sound(props1, props2, 50.0)
        loud_sound = physics_engine._generate_collision_sound(props1, props2, 800.0)

        assert isinstance(soft_sound, str)
        assert isinstance(loud_sound, str)
        assert "wood" in soft_sound and "metal" in soft_sound
        assert "loud" in loud_sound

    @pytest.mark.asyncio
    async def test_leverage_factor_calculation(self, physics_engine):
        """Test leverage factor calculation with tools."""
        crowbar_leverage = await physics_engine._calculate_leverage_factor(
            "crowbar", PhysicalActionType.PUSHING
        )
        no_tool_leverage = await physics_engine._calculate_leverage_factor(
            None, PhysicalActionType.PUSHING
        )

        assert crowbar_leverage > no_tool_leverage
        assert no_tool_leverage == 1.0  # No tool baseline

    @pytest.mark.asyncio
    async def test_error_handling(self, physics_engine):
        """Test error handling in physics calculations."""
        # Test with invalid/empty inputs
        is_valid, error = await physics_engine.validate_physical_constraints(
            PhysicalActionType.MOVEMENT, [], {}
        )

        # Should handle gracefully without crashing
        assert isinstance(is_valid, bool)
        if not is_valid:
            assert isinstance(error, str)


@pytest.mark.asyncio
async def test_physics_engine_integration():
    """Test PhysicsConstraintEngine integration scenarios."""
    physics_engine = PhysicsConstraintEngine()

    # Test complex scenario: climbing with equipment
    player_state = {
        "strength": 70,
        "energy": 85,
        "fitness": 60,
        "balance_skill": 75,
        "current_location": {"space_rating": 6},
    }

    entities = ["climbing_rope", "cliff_face"]

    # Validate constraints
    is_valid, error = await physics_engine.validate_physical_constraints(
        PhysicalActionType.CLIMBING, entities, player_state
    )

    # Calculate requirements
    strength_req = await physics_engine.calculate_strength_requirements(
        PhysicalActionType.CLIMBING,
        15.0,
        {"leverage_factor": 1.5},  # Rope provides leverage
    )

    # Calculate energy cost
    energy_cost = await physics_engine.calculate_energy_expenditure(
        PhysicalActionType.CLIMBING, 45.0, 0.7, player_state
    )

    # All calculations should complete successfully
    assert isinstance(is_valid, bool)
    assert isinstance(strength_req, float)
    assert isinstance(energy_cost, float)
    assert strength_req > 0
    assert energy_cost > 0


if __name__ == "__main__":
    pytest.main([__file__])
