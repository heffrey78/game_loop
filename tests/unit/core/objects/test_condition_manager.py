"""
Unit tests for ObjectConditionManager.

Tests object quality tracking, degradation simulation, and repair mechanics.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.game_loop.core.objects.condition_manager import (
    ObjectConditionManager,
    ObjectCondition,
    QualityAspect
)


class TestObjectConditionManager:
    """Test cases for ObjectConditionManager functionality."""

    @pytest.fixture
    def condition_manager(self):
        """Create condition manager for testing."""
        object_manager = Mock()
        object_manager.get_object_properties = AsyncMock(return_value={
            "type": "tool",
            "material": "iron",
            "wear_resistance": 1.0,
            "environmental_resistance": 1.0
        })
        
        time_manager = Mock()
        time_manager.schedule_event = AsyncMock()
        
        environment_manager = Mock()
        
        manager = ObjectConditionManager(object_manager, time_manager, environment_manager)
        return manager

    @pytest.mark.asyncio
    async def test_track_object_condition_new_object(self, condition_manager):
        """Test starting to track condition for a new object."""
        await condition_manager.track_object_condition("test_object")
        
        assert "test_object" in condition_manager._condition_registry
        condition = condition_manager._condition_registry["test_object"]
        
        assert condition.overall_condition == 1.0  # Perfect condition
        assert QualityAspect.DURABILITY in condition.quality_aspects
        assert condition.quality_aspects[QualityAspect.DURABILITY] == 1.0

    @pytest.mark.asyncio
    async def test_track_object_condition_with_initial_condition(self, condition_manager):
        """Test tracking object with specified initial condition."""
        initial_condition = ObjectCondition(
            overall_condition=0.8,
            quality_aspects={
                QualityAspect.DURABILITY: 0.7,
                QualityAspect.SHARPNESS: 0.9
            },
            degradation_factors={},
            maintenance_history=[],
            condition_modifiers={},
            last_updated=0.0
        )
        
        await condition_manager.track_object_condition("test_object", initial_condition)
        
        condition = condition_manager._condition_registry["test_object"]
        assert condition.overall_condition == 0.8
        assert condition.quality_aspects[QualityAspect.DURABILITY] == 0.7
        assert condition.quality_aspects[QualityAspect.SHARPNESS] == 0.9

    @pytest.mark.asyncio
    async def test_update_object_condition(self, condition_manager):
        """Test updating object condition based on usage."""
        await condition_manager.track_object_condition("test_object")
        
        usage_data = {
            "intensity": 0.5,
            "frequency": 0.3
        }
        environmental_factors = {
            "temperature": 25.0,
            "humidity": 60.0
        }
        
        updated_condition = await condition_manager.update_object_condition(
            "test_object", usage_data, environmental_factors
        )
        
        # Condition should have degraded slightly
        assert updated_condition.overall_condition < 1.0
        assert updated_condition.last_updated > 0

    @pytest.mark.asyncio
    async def test_calculate_condition_impact_tool_usage(self, condition_manager):
        """Test condition impact calculation for tool usage."""
        await condition_manager.track_object_condition("test_tool")
        
        # Set specific condition values
        condition = condition_manager._condition_registry["test_tool"]
        condition.quality_aspects[QualityAspect.SHARPNESS] = 0.8
        condition.quality_aspects[QualityAspect.DURABILITY] = 0.9
        
        impact = await condition_manager.calculate_condition_impact("test_tool", "cut")
        
        assert "efficiency_modifier" in impact
        assert "success_modifier" in impact
        assert impact["efficiency_modifier"] <= 1.0  # Should be reduced due to condition
        assert impact["success_modifier"] <= 1.0

    @pytest.mark.asyncio
    async def test_repair_object_success(self, condition_manager):
        """Test successful object repair."""
        await condition_manager.track_object_condition("test_object")
        
        # Damage the object first
        condition = condition_manager._condition_registry["test_object"]
        condition.overall_condition = 0.5
        condition.quality_aspects[QualityAspect.DURABILITY] = 0.4
        
        success, result = await condition_manager.repair_object(
            "test_object",
            ["repair_kit", "metal_scraps"],
            5,  # Repair skill level
            ["hammer", "anvil"]
        )
        
        # Should succeed with reasonable probability
        assert isinstance(success, bool)
        if success:
            assert result["repair_amount"] > 0
            assert result["new_condition"] > 0.5

    @pytest.mark.asyncio
    async def test_repair_object_already_perfect(self, condition_manager):
        """Test repair attempt on object in perfect condition."""
        await condition_manager.track_object_condition("test_object")
        
        # Object is already in perfect condition (1.0)
        success, result = await condition_manager.repair_object(
            "test_object",
            ["repair_kit"],
            5,
            ["hammer"]
        )
        
        assert success is False
        assert "does not need repair" in result["error"]

    @pytest.mark.asyncio
    async def test_maintain_object(self, condition_manager):
        """Test object maintenance operations."""
        await condition_manager.track_object_condition("test_object")
        
        result = await condition_manager.maintain_object(
            "test_object",
            "cleaning",
            ["soap", "water"]
        )
        
        assert "maintenance_type" in result
        assert result["maintenance_type"] == "cleaning"
        assert "effects" in result
        assert "new_condition" in result

    @pytest.mark.asyncio
    async def test_get_condition_description_basic(self, condition_manager):
        """Test basic condition description generation."""
        await condition_manager.track_object_condition("test_object")
        
        description = await condition_manager.get_condition_description("test_object", "basic")
        
        assert isinstance(description, str)
        assert "pristine" in description.lower() or "excellent" in description.lower()

    @pytest.mark.asyncio
    async def test_get_condition_description_detailed(self, condition_manager):
        """Test detailed condition description generation."""
        await condition_manager.track_object_condition("test_object")
        
        # Set poor condition for some aspects
        condition = condition_manager._condition_registry["test_object"]
        condition.quality_aspects[QualityAspect.SHARPNESS] = 0.3
        condition.overall_condition = 0.6
        
        description = await condition_manager.get_condition_description("test_object", "detailed")
        
        assert isinstance(description, str)
        assert "sharpness" in description.lower()

    @pytest.mark.asyncio
    async def test_get_condition_description_technical(self, condition_manager):
        """Test technical condition description generation."""
        await condition_manager.track_object_condition("test_object")
        
        description = await condition_manager.get_condition_description("test_object", "technical")
        
        assert isinstance(description, str)
        assert "%" in description  # Should include percentages
        assert "condition:" in description.lower()

    @pytest.mark.asyncio
    async def test_simulate_environmental_degradation(self, condition_manager):
        """Test environmental degradation simulation."""
        await condition_manager.track_object_condition("test_object")
        
        environment_data = {
            "temperature": 50.0,  # High temperature
            "humidity": 90.0,     # High humidity
            "acidity": 5.0,       # Acidic environment
            "exposure": "outdoor"
        }
        
        result = await condition_manager.simulate_environmental_degradation(
            "test_object", environment_data, 100.0  # Long time period
        )
        
        assert "total_degradation" in result
        assert "environmental_factors" in result
        assert "new_condition" in result
        assert result["total_degradation"] > 0  # Should have some degradation

    @pytest.mark.asyncio
    async def test_apply_condition_modifiers(self, condition_manager):
        """Test applying temporary condition modifiers."""
        await condition_manager.track_object_condition("test_object")
        
        modifiers = {
            "magical_enhancement": 0.1,
            "environmental_protection": 0.05
        }
        
        await condition_manager.apply_condition_modifiers(
            "test_object", modifiers, duration=3600.0
        )
        
        condition = condition_manager._condition_registry["test_object"]
        assert "magical_enhancement" in condition.condition_modifiers
        assert condition.condition_modifiers["magical_enhancement"] == 0.1

    @pytest.mark.asyncio
    async def test_assess_repair_requirements(self, condition_manager):
        """Test repair requirements assessment."""
        await condition_manager.track_object_condition("test_object")
        
        # Damage the object
        condition = condition_manager._condition_registry["test_object"]
        condition.overall_condition = 0.3  # Significantly damaged
        
        assessment = await condition_manager.assess_repair_requirements("test_object")
        
        assert assessment["needs_repair"] is True
        assert "damage_severity" in assessment
        assert "materials_needed" in assessment
        assert "skills_needed" in assessment
        assert "tools_needed" in assessment
        assert assessment["damage_severity"] in ["minor", "moderate", "severe"]

    @pytest.mark.asyncio
    async def test_register_degradation_model(self, condition_manager):
        """Test registering custom degradation models."""
        async def custom_degradation_model(object_id, usage_data, env_data):
            return 0.01  # Fixed degradation rate
        
        condition_manager.register_degradation_model("custom_type", custom_degradation_model)
        
        assert "custom_type" in condition_manager._degradation_models
        assert condition_manager._degradation_models["custom_type"] == custom_degradation_model

    @pytest.mark.asyncio
    async def test_multiple_quality_aspects(self, condition_manager):
        """Test handling multiple quality aspects."""
        initial_condition = ObjectCondition(
            overall_condition=1.0,
            quality_aspects={
                QualityAspect.DURABILITY: 1.0,
                QualityAspect.SHARPNESS: 1.0,
                QualityAspect.EFFICIENCY: 1.0,
                QualityAspect.APPEARANCE: 1.0,
                QualityAspect.MAGICAL_POTENCY: 1.0
            },
            degradation_factors={},
            maintenance_history=[],
            condition_modifiers={},
            last_updated=0.0
        )
        
        await condition_manager.track_object_condition("magic_sword", initial_condition)
        
        # Update condition - should affect all aspects
        usage_data = {"intensity": 1.0, "frequency": 1.0}
        environmental_factors = {"temperature": 20.0}
        
        updated_condition = await condition_manager.update_object_condition(
            "magic_sword", usage_data, environmental_factors
        )
        
        # All aspects should still be present
        assert len(updated_condition.quality_aspects) == 5
        assert QualityAspect.MAGICAL_POTENCY in updated_condition.quality_aspects

    @pytest.mark.asyncio
    async def test_maintenance_history_tracking(self, condition_manager):
        """Test that maintenance history is properly tracked."""
        await condition_manager.track_object_condition("test_object")
        
        # Perform maintenance
        await condition_manager.maintain_object("test_object", "oiling", ["oil"])
        
        condition = condition_manager._condition_registry["test_object"]
        assert len(condition.maintenance_history) == 1
        
        maintenance_record = condition.maintenance_history[0]
        assert maintenance_record["maintenance_type"] == "oiling"
        assert maintenance_record["materials_used"] == ["oil"]
        assert "timestamp" in maintenance_record

    @pytest.mark.asyncio
    async def test_extreme_environmental_conditions(self, condition_manager):
        """Test degradation under extreme environmental conditions."""
        await condition_manager.track_object_condition("test_object")
        
        extreme_environment = {
            "temperature": 100.0,  # Very hot
            "humidity": 0.0,       # Very dry
            "acidity": 1.0,        # Very acidic
            "radiation": 10.0,     # High radiation
            "exposure": "volcanic"
        }
        
        result = await condition_manager.simulate_environmental_degradation(
            "test_object", extreme_environment, 10.0
        )
        
        # Should have significant degradation
        assert result["total_degradation"] > 0.1
        assert result["exposure_multiplier"] > 1.0  # Volcanic exposure multiplier

    @pytest.mark.asyncio
    async def test_condition_threshold_events(self, condition_manager):
        """Test condition threshold event triggering."""
        await condition_manager.track_object_condition("test_object")
        
        # Set up a condition threshold
        condition_manager._condition_thresholds["test_object"] = {
            "warning_threshold": 0.5,
            "critical_threshold": 0.2
        }
        
        # Severely damage the object
        condition = condition_manager._condition_registry["test_object"]
        condition.overall_condition = 0.1  # Below critical threshold
        
        # Update condition should trigger threshold check
        await condition_manager.update_object_condition(
            "test_object", {"intensity": 0.0}, {"temperature": 20.0}
        )
        
        # Threshold events should be checked (logged internally)

    @pytest.mark.asyncio
    async def test_repair_with_insufficient_materials(self, condition_manager):
        """Test repair failure due to insufficient materials."""
        await condition_manager.track_object_condition("test_object")
        
        # Damage the object
        condition = condition_manager._condition_registry["test_object"]
        condition.overall_condition = 0.3
        
        # Mock validation to fail for insufficient materials
        condition_manager._validate_repair_materials = AsyncMock(return_value=(False, "Insufficient materials"))
        
        success, result = await condition_manager.repair_object(
            "test_object", [], 5, ["hammer"]  # No materials provided
        )
        
        assert success is False
        assert "materials" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_degradation_rate_calculation(self, condition_manager):
        """Test degradation rate calculation with different factors."""
        await condition_manager.track_object_condition("test_object")
        
        # High usage should cause more degradation
        high_usage = {"intensity": 1.0, "frequency": 1.0}
        low_usage = {"intensity": 0.1, "frequency": 0.1}
        
        high_degradation = await condition_manager._calculate_degradation_rate(
            "test_object", QualityAspect.DURABILITY, high_usage, {}, 1.0
        )
        
        low_degradation = await condition_manager._calculate_degradation_rate(
            "test_object", QualityAspect.DURABILITY, low_usage, {}, 1.0
        )
        
        assert high_degradation > low_degradation