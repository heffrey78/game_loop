"""
Object Condition Manager for quality and durability tracking.

This module provides comprehensive object condition tracking including multi-dimensional
quality aspects, environmental degradation simulation, and repair mechanics.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QualityAspect(Enum):
    """Different aspects of object quality that can be tracked."""
    
    DURABILITY = "durability"
    SHARPNESS = "sharpness"
    EFFICIENCY = "efficiency"
    APPEARANCE = "appearance"
    MAGICAL_POTENCY = "magical_potency"
    PURITY = "purity"
    STABILITY = "stability"


@dataclass
class ObjectCondition:
    """Comprehensive condition information for an object."""
    
    overall_condition: float  # 0.0 to 1.0
    quality_aspects: Dict[QualityAspect, float]
    degradation_factors: Dict[str, float]
    maintenance_history: List[Dict[str, Any]]
    condition_modifiers: Dict[str, float]
    last_updated: float


class ObjectConditionManager:
    """
    Track and manage object quality, durability, and condition over time.
    
    This class provides comprehensive condition management including:
    - Multi-dimensional quality tracking
    - Environmental degradation simulation
    - Repair and maintenance mechanics
    - Condition-based interaction modifications
    - Quality impact on functionality
    """

    def __init__(
        self, 
        object_manager: Any = None, 
        time_manager: Any = None, 
        environment_manager: Any = None
    ):
        """
        Initialize the object condition manager.

        Args:
            object_manager: Manager for object data and properties
            time_manager: Manager for game time and scheduling
            environment_manager: Manager for environmental factors
        """
        self.objects = object_manager
        self.time = time_manager
        self.environment = environment_manager
        self._condition_registry: Dict[str, ObjectCondition] = {}
        self._degradation_models: Dict[str, Callable] = {}
        self._repair_recipes: Dict[str, Dict[str, Any]] = {}
        self._maintenance_schedules: Dict[str, Dict[str, Any]] = {}
        self._condition_thresholds: Dict[str, Dict[str, float]] = {}
        self._quality_modifiers: Dict[str, Callable] = {}
        self._initialize_degradation_models()

    async def track_object_condition(
        self, 
        object_id: str, 
        initial_condition: Optional[ObjectCondition] = None
    ) -> None:
        """
        Begin tracking condition for an object.

        Args:
            object_id: Object to start tracking
            initial_condition: Initial condition state (optional)
        """
        try:
            if object_id in self._condition_registry:
                logger.warning(f"Object {object_id} is already being tracked")
                return
            
            if initial_condition:
                condition = initial_condition
            else:
                # Create default perfect condition
                condition = ObjectCondition(
                    overall_condition=1.0,
                    quality_aspects={
                        QualityAspect.DURABILITY: 1.0,
                        QualityAspect.EFFICIENCY: 1.0,
                        QualityAspect.APPEARANCE: 1.0,
                    },
                    degradation_factors={},
                    maintenance_history=[],
                    condition_modifiers={},
                    last_updated=asyncio.get_event_loop().time(),
                )
            
            self._condition_registry[object_id] = condition
            
            # Initialize object-specific degradation factors
            await self._initialize_object_degradation_factors(object_id)
            
            logger.info(f"Started tracking condition for object {object_id}")

        except Exception as e:
            logger.error(f"Error tracking object condition: {e}")

    async def update_object_condition(
        self, 
        object_id: str, 
        usage_data: Dict[str, Any],
        environmental_factors: Dict[str, Any]
    ) -> ObjectCondition:
        """
        Update object condition based on usage and environment.

        Args:
            object_id: Object to update
            usage_data: Information about object usage
            environmental_factors: Current environmental conditions

        Returns:
            Updated ObjectCondition
        """
        try:
            if object_id not in self._condition_registry:
                await self.track_object_condition(object_id)
            
            condition = self._condition_registry[object_id]
            current_time = asyncio.get_event_loop().time()
            time_elapsed = current_time - condition.last_updated
            
            # Calculate degradation for each quality aspect
            for aspect in condition.quality_aspects:
                degradation_rate = await self._calculate_degradation_rate(
                    object_id, aspect, usage_data, environmental_factors, time_elapsed
                )
                
                # Apply degradation
                current_value = condition.quality_aspects[aspect]
                new_value = max(0.0, current_value - degradation_rate)
                condition.quality_aspects[aspect] = new_value
            
            # Update overall condition (weighted average of aspects)
            condition.overall_condition = await self._calculate_overall_condition(
                object_id, condition.quality_aspects
            )
            
            # Apply condition modifiers
            for modifier_name, modifier_value in condition.condition_modifiers.items():
                condition.overall_condition = max(0.0, min(1.0, 
                    condition.overall_condition + modifier_value
                ))
            
            # Update degradation factors based on current state
            await self._update_degradation_factors(object_id, condition, environmental_factors)
            
            # Check for condition thresholds and trigger events
            await self._check_condition_thresholds(object_id, condition)
            
            condition.last_updated = current_time
            
            return condition

        except Exception as e:
            logger.error(f"Error updating object condition: {e}")
            return self._condition_registry.get(object_id, ObjectCondition(
                overall_condition=1.0, quality_aspects={}, degradation_factors={},
                maintenance_history=[], condition_modifiers={}, last_updated=0.0
            ))

    async def calculate_condition_impact(
        self, 
        object_id: str, 
        interaction_type: str
    ) -> Dict[str, float]:
        """
        Calculate how object condition affects interaction outcomes.

        Args:
            object_id: Object being used
            interaction_type: Type of interaction

        Returns:
            Dict with condition impact factors
        """
        try:
            if object_id not in self._condition_registry:
                return {"efficiency_modifier": 1.0, "success_modifier": 1.0}
            
            condition = self._condition_registry[object_id]
            
            # Base condition impact
            condition_factor = condition.overall_condition
            
            # Interaction-specific impacts
            impact_factors = {
                "efficiency_modifier": 1.0,
                "success_modifier": 1.0,
                "energy_cost_modifier": 1.0,
                "time_modifier": 1.0,
            }
            
            # Tools and weapons are heavily affected by sharpness and durability
            if interaction_type in ["cut", "chop", "slice", "weapon_attack"]:
                sharpness = condition.quality_aspects.get(QualityAspect.SHARPNESS, 1.0)
                durability = condition.quality_aspects.get(QualityAspect.DURABILITY, 1.0)
                
                impact_factors["efficiency_modifier"] = (sharpness + durability) / 2.0
                impact_factors["success_modifier"] = max(0.1, sharpness * 0.8 + durability * 0.2)
            
            # Magical items are affected by magical potency
            elif interaction_type in ["cast_spell", "enchant", "magical_effect"]:
                potency = condition.quality_aspects.get(QualityAspect.MAGICAL_POTENCY, 1.0)
                stability = condition.quality_aspects.get(QualityAspect.STABILITY, 1.0)
                
                impact_factors["efficiency_modifier"] = potency
                impact_factors["success_modifier"] = stability
            
            # General tools affected by efficiency and durability
            elif interaction_type in ["repair", "craft", "build"]:
                efficiency = condition.quality_aspects.get(QualityAspect.EFFICIENCY, 1.0)
                durability = condition.quality_aspects.get(QualityAspect.DURABILITY, 1.0)
                
                impact_factors["efficiency_modifier"] = efficiency
                impact_factors["time_modifier"] = 2.0 - efficiency  # Lower efficiency = more time
            
            # Apply overall condition as a general modifier
            for key in impact_factors:
                if key != "energy_cost_modifier":  # Energy cost increases with poor condition
                    impact_factors[key] *= (0.5 + 0.5 * condition_factor)
                else:
                    impact_factors[key] = 1.0 + (1.0 - condition_factor) * 0.5
            
            return impact_factors

        except Exception as e:
            logger.error(f"Error calculating condition impact: {e}")
            return {"efficiency_modifier": 1.0, "success_modifier": 1.0}

    async def repair_object(
        self, 
        object_id: str, 
        repair_materials: List[str],
        repair_skill: int, 
        repair_tools: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Attempt to repair object using materials and skills.

        Args:
            object_id: Object to repair
            repair_materials: Materials used for repair
            repair_skill: Skill level of the repairer
            repair_tools: Tools used for repair

        Returns:
            Tuple of (success, repair_result)
        """
        try:
            if object_id not in self._condition_registry:
                return False, {"error": f"Object {object_id} not being tracked"}
            
            condition = self._condition_registry[object_id]
            
            # Check if repair is needed
            if condition.overall_condition >= 1.0:
                return False, {"error": "Object does not need repair"}
            
            # Get repair requirements
            repair_requirements = await self._get_repair_requirements(object_id)
            
            # Validate materials
            materials_valid, materials_error = await self._validate_repair_materials(
                repair_requirements, repair_materials
            )
            if not materials_valid:
                return False, {"error": materials_error}
            
            # Validate tools
            tools_valid, tools_error = await self._validate_repair_tools(
                repair_requirements, repair_tools
            )
            if not tools_valid:
                return False, {"error": tools_error}
            
            # Calculate repair success probability
            success_probability = await self._calculate_repair_success_probability(
                object_id, repair_skill, repair_materials, repair_tools
            )
            
            # Determine success
            import random
            repair_successful = random.random() < success_probability
            
            if repair_successful:
                # Apply repair effects
                repair_amount = await self._calculate_repair_amount(
                    object_id, repair_skill, repair_materials, repair_tools
                )
                
                # Improve condition aspects
                for aspect in condition.quality_aspects:
                    current_value = condition.quality_aspects[aspect]
                    aspect_repair = repair_amount * self._get_aspect_repair_factor(aspect)
                    condition.quality_aspects[aspect] = min(1.0, current_value + aspect_repair)
                
                # Recalculate overall condition
                condition.overall_condition = await self._calculate_overall_condition(
                    object_id, condition.quality_aspects
                )
                
                # Record repair in history
                repair_record = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "repair_amount": repair_amount,
                    "materials_used": repair_materials,
                    "tools_used": repair_tools,
                    "repairer_skill": repair_skill,
                    "success": True,
                }
                condition.maintenance_history.append(repair_record)
                
                return True, {
                    "repair_amount": repair_amount,
                    "new_condition": condition.overall_condition,
                    "materials_consumed": repair_materials,
                    "repair_quality": "excellent" if repair_amount > 0.3 else "good" if repair_amount > 0.15 else "fair",
                }
            else:
                # Partial repair or failure
                partial_repair = repair_amount * 0.3  # 30% of full repair on failure
                
                for aspect in condition.quality_aspects:
                    current_value = condition.quality_aspects[aspect]
                    aspect_repair = partial_repair * self._get_aspect_repair_factor(aspect)
                    condition.quality_aspects[aspect] = min(1.0, current_value + aspect_repair)
                
                condition.overall_condition = await self._calculate_overall_condition(
                    object_id, condition.quality_aspects
                )
                
                # Record failed repair
                repair_record = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "repair_amount": partial_repair,
                    "materials_used": repair_materials,
                    "tools_used": repair_tools,
                    "repairer_skill": repair_skill,
                    "success": False,
                }
                condition.maintenance_history.append(repair_record)
                
                return False, {
                    "error": "Repair failed",
                    "partial_repair": partial_repair,
                    "new_condition": condition.overall_condition,
                    "materials_consumed": repair_materials,  # Materials still consumed
                }

        except Exception as e:
            logger.error(f"Error repairing object: {e}")
            return False, {"error": str(e)}

    async def maintain_object(
        self, 
        object_id: str, 
        maintenance_type: str,
        maintenance_materials: List[str]
    ) -> Dict[str, Any]:
        """
        Perform maintenance to slow degradation.

        Args:
            object_id: Object to maintain
            maintenance_type: Type of maintenance
            maintenance_materials: Materials used for maintenance

        Returns:
            Dict with maintenance results
        """
        try:
            if object_id not in self._condition_registry:
                return {"error": f"Object {object_id} not being tracked"}
            
            condition = self._condition_registry[object_id]
            
            maintenance_effects = {
                "cleaning": {
                    "appearance_bonus": 0.05,
                    "degradation_reduction": 0.1,
                    "duration": 3600.0,  # 1 hour
                },
                "oiling": {
                    "efficiency_bonus": 0.03,
                    "degradation_reduction": 0.15,
                    "duration": 7200.0,  # 2 hours
                },
                "sharpening": {
                    "sharpness_bonus": 0.1,
                    "degradation_reduction": 0.05,
                    "duration": 1800.0,  # 30 minutes
                },
                "calibration": {
                    "efficiency_bonus": 0.05,
                    "stability_bonus": 0.05,
                    "duration": 3600.0,
                },
            }
            
            if maintenance_type not in maintenance_effects:
                return {"error": f"Unknown maintenance type: {maintenance_type}"}
            
            effects = maintenance_effects[maintenance_type]
            
            # Apply immediate benefits
            for aspect in QualityAspect:
                bonus_key = f"{aspect.value}_bonus"
                if bonus_key in effects:
                    current_value = condition.quality_aspects.get(aspect, 1.0)
                    bonus = effects[bonus_key]
                    condition.quality_aspects[aspect] = min(1.0, current_value + bonus)
            
            # Apply temporary condition modifiers
            modifier_name = f"maintenance_{maintenance_type}"
            condition.condition_modifiers[modifier_name] = effects.get("degradation_reduction", 0.0)
            
            # Schedule modifier removal
            if self.time:
                await self.time.schedule_event(
                    asyncio.get_event_loop().time() + effects.get("duration", 3600.0),
                    "remove_condition_modifier",
                    {"object_id": object_id, "modifier_name": modifier_name}
                )
            
            # Record maintenance
            maintenance_record = {
                "timestamp": asyncio.get_event_loop().time(),
                "maintenance_type": maintenance_type,
                "materials_used": maintenance_materials,
                "effects_applied": effects,
            }
            condition.maintenance_history.append(maintenance_record)
            
            return {
                "maintenance_type": maintenance_type,
                "effects": effects,
                "new_condition": condition.overall_condition,
                "duration": effects.get("duration", 3600.0),
            }

        except Exception as e:
            logger.error(f"Error maintaining object: {e}")
            return {"error": str(e)}

    async def get_condition_description(
        self, 
        object_id: str, 
        detail_level: str = "basic"
    ) -> str:
        """
        Get human-readable description of object condition.

        Args:
            object_id: Object to describe
            detail_level: Level of detail (basic, detailed, technical)

        Returns:
            Human-readable condition description
        """
        try:
            if object_id not in self._condition_registry:
                return "Condition unknown"
            
            condition = self._condition_registry[object_id]
            overall = condition.overall_condition
            
            # Basic condition descriptions
            if overall >= 0.95:
                base_desc = "pristine"
            elif overall >= 0.85:
                base_desc = "excellent"
            elif overall >= 0.7:
                base_desc = "good"
            elif overall >= 0.5:
                base_desc = "fair"
            elif overall >= 0.3:
                base_desc = "poor"
            elif overall >= 0.1:
                base_desc = "damaged"
            else:
                base_desc = "severely damaged"
            
            if detail_level == "basic":
                return f"The object is in {base_desc} condition."
            
            elif detail_level == "detailed":
                details = []
                
                # Add aspect-specific details
                for aspect, value in condition.quality_aspects.items():
                    if value < 0.5:  # Only mention poor aspects
                        aspect_name = aspect.value.replace("_", " ")
                        if value < 0.2:
                            details.append(f"very poor {aspect_name}")
                        else:
                            details.append(f"poor {aspect_name}")
                
                if details:
                    return f"The object is in {base_desc} condition with {', '.join(details)}."
                else:
                    return f"The object is in {base_desc} condition with no major issues."
            
            elif detail_level == "technical":
                aspects_desc = []
                for aspect, value in condition.quality_aspects.items():
                    aspect_name = aspect.value.replace("_", " ")
                    percentage = int(value * 100)
                    aspects_desc.append(f"{aspect_name}: {percentage}%")
                
                modifiers_desc = ""
                if condition.condition_modifiers:
                    mod_list = [f"{name}: {value:+.2f}" for name, value in condition.condition_modifiers.items()]
                    modifiers_desc = f" (modifiers: {', '.join(mod_list)})"
                
                return f"Overall condition: {int(overall * 100)}%. {', '.join(aspects_desc)}.{modifiers_desc}"
            
            return f"The object is in {base_desc} condition."

        except Exception as e:
            logger.error(f"Error getting condition description: {e}")
            return "Condition assessment failed"

    async def simulate_environmental_degradation(
        self, 
        object_id: str, 
        environment_data: Dict[str, Any],
        time_elapsed: float
    ) -> Dict[str, Any]:
        """
        Simulate degradation due to environmental factors.

        Args:
            object_id: Object to simulate
            environment_data: Environmental conditions
            time_elapsed: Time period to simulate

        Returns:
            Dict with degradation simulation results
        """
        try:
            if object_id not in self._condition_registry:
                return {"error": f"Object {object_id} not being tracked"}
            
            condition = self._condition_registry[object_id]
            
            # Extract environmental factors
            temperature = environment_data.get("temperature", 20.0)  # Celsius
            humidity = environment_data.get("humidity", 50.0)        # Percentage
            acidity = environment_data.get("acidity", 7.0)           # pH
            radiation = environment_data.get("radiation", 0.0)       # Radiation level
            exposure = environment_data.get("exposure", "indoor")    # indoor/outdoor/underwater/etc
            
            degradation_results = {}
            
            # Temperature effects
            temperature_stress = abs(temperature - 20.0) / 100.0  # Normalized stress
            if temperature_stress > 0.1:
                temp_degradation = temperature_stress * time_elapsed * 0.01
                degradation_results["temperature"] = temp_degradation
            
            # Humidity effects
            if humidity > 80.0 or humidity < 20.0:
                humidity_stress = abs(humidity - 50.0) / 100.0
                humidity_degradation = humidity_stress * time_elapsed * 0.005
                degradation_results["humidity"] = humidity_degradation
            
            # Acidity effects
            if acidity < 6.0 or acidity > 8.0:
                acid_stress = abs(acidity - 7.0) / 10.0
                acid_degradation = acid_stress * time_elapsed * 0.02
                degradation_results["acidity"] = acid_degradation
            
            # Radiation effects
            if radiation > 0.0:
                radiation_degradation = radiation * time_elapsed * 0.001
                degradation_results["radiation"] = radiation_degradation
            
            # Exposure effects
            exposure_multipliers = {
                "indoor": 1.0,
                "outdoor": 1.5,
                "underwater": 2.0,
                "volcanic": 3.0,
                "space": 4.0,
            }
            exposure_multiplier = exposure_multipliers.get(exposure, 1.0)
            
            # Apply all degradation
            total_degradation = sum(degradation_results.values()) * exposure_multiplier
            
            if total_degradation > 0:
                # Apply proportionally to all quality aspects
                for aspect in condition.quality_aspects:
                    current_value = condition.quality_aspects[aspect]
                    aspect_degradation = total_degradation * self._get_aspect_vulnerability(aspect, environment_data)
                    condition.quality_aspects[aspect] = max(0.0, current_value - aspect_degradation)
                
                # Recalculate overall condition
                condition.overall_condition = await self._calculate_overall_condition(
                    object_id, condition.quality_aspects
                )
            
            return {
                "total_degradation": total_degradation,
                "environmental_factors": degradation_results,
                "exposure_multiplier": exposure_multiplier,
                "new_condition": condition.overall_condition,
            }

        except Exception as e:
            logger.error(f"Error simulating environmental degradation: {e}")
            return {"error": str(e)}

    async def apply_condition_modifiers(
        self, 
        object_id: str, 
        modifiers: Dict[str, float],
        duration: Optional[float] = None
    ) -> None:
        """
        Apply temporary or permanent condition modifiers.

        Args:
            object_id: Object to modify
            modifiers: Modifiers to apply
            duration: Duration of modifiers (None = permanent)
        """
        try:
            if object_id not in self._condition_registry:
                await self.track_object_condition(object_id)
            
            condition = self._condition_registry[object_id]
            
            for modifier_name, modifier_value in modifiers.items():
                condition.condition_modifiers[modifier_name] = modifier_value
                
                # Schedule removal if temporary
                if duration is not None and self.time:
                    await self.time.schedule_event(
                        asyncio.get_event_loop().time() + duration,
                        "remove_condition_modifier",
                        {"object_id": object_id, "modifier_name": modifier_name}
                    )

        except Exception as e:
            logger.error(f"Error applying condition modifiers: {e}")

    async def assess_repair_requirements(self, object_id: str) -> Dict[str, Any]:
        """
        Assess what would be needed to fully repair an object.

        Args:
            object_id: Object to assess

        Returns:
            Dict with repair requirements assessment
        """
        try:
            if object_id not in self._condition_registry:
                return {"error": f"Object {object_id} not being tracked"}
            
            condition = self._condition_registry[object_id]
            
            if condition.overall_condition >= 1.0:
                return {"needs_repair": False, "message": "Object is in perfect condition"}
            
            # Calculate repair requirements based on damage
            damage_amount = 1.0 - condition.overall_condition
            
            # Estimate materials needed
            material_requirements = await self._estimate_repair_materials(object_id, damage_amount)
            
            # Estimate skill requirements
            skill_requirements = await self._estimate_repair_skills(object_id, damage_amount)
            
            # Estimate tool requirements
            tool_requirements = await self._estimate_repair_tools(object_id, damage_amount)
            
            # Estimate time and cost
            repair_time = damage_amount * 3600.0  # Base 1 hour per full repair
            repair_cost = damage_amount * 100.0   # Base cost scaling
            
            return {
                "needs_repair": True,
                "damage_severity": "severe" if damage_amount > 0.7 else "moderate" if damage_amount > 0.3 else "minor",
                "materials_needed": material_requirements,
                "skills_needed": skill_requirements,
                "tools_needed": tool_requirements,
                "estimated_time": repair_time,
                "estimated_cost": repair_cost,
                "repair_difficulty": damage_amount,
            }

        except Exception as e:
            logger.error(f"Error assessing repair requirements: {e}")
            return {"error": str(e)}

    def register_degradation_model(self, object_type: str, model: Callable) -> None:
        """
        Register custom degradation model for object type.

        Args:
            object_type: Type of object
            model: Degradation model function
        """
        self._degradation_models[object_type] = model

    # Private helper methods

    async def _calculate_degradation_rate(
        self, 
        object_id: str, 
        aspect: QualityAspect, 
        usage_data: Dict[str, Any], 
        environmental_factors: Dict[str, Any], 
        time_elapsed: float
    ) -> float:
        """Calculate degradation rate for a specific quality aspect."""
        try:
            # Base degradation rate
            base_rate = 0.001  # 0.1% per time unit
            
            # Usage-based degradation
            usage_intensity = usage_data.get("intensity", 0.0)
            usage_frequency = usage_data.get("frequency", 0.0)
            usage_degradation = base_rate * usage_intensity * usage_frequency
            
            # Environmental degradation
            environmental_stress = self._calculate_environmental_stress(environmental_factors)
            environmental_degradation = base_rate * environmental_stress
            
            # Aspect-specific factors
            aspect_multiplier = self._get_aspect_degradation_multiplier(aspect, usage_data)
            
            # Total degradation
            total_degradation = (usage_degradation + environmental_degradation) * aspect_multiplier * time_elapsed
            
            return total_degradation

        except Exception:
            return 0.001  # Default minimal degradation

    async def _calculate_overall_condition(
        self, 
        object_id: str, 
        quality_aspects: Dict[QualityAspect, float]
    ) -> float:
        """Calculate overall condition from quality aspects."""
        if not quality_aspects:
            return 1.0
        
        # Get object properties to determine aspect weights
        object_properties = await self._get_object_properties(object_id)
        
        # Default weights
        weights = {
            QualityAspect.DURABILITY: 0.4,
            QualityAspect.EFFICIENCY: 0.3,
            QualityAspect.APPEARANCE: 0.1,
            QualityAspect.SHARPNESS: 0.2,
            QualityAspect.MAGICAL_POTENCY: 0.3,
            QualityAspect.PURITY: 0.2,
            QualityAspect.STABILITY: 0.2,
        }
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for aspect, value in quality_aspects.items():
            weight = weights.get(aspect, 0.1)
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0

    async def _initialize_object_degradation_factors(self, object_id: str) -> None:
        """Initialize object-specific degradation factors."""
        try:
            condition = self._condition_registry[object_id]
            object_properties = await self._get_object_properties(object_id)
            
            if object_properties:
                # Set initial degradation factors based on object type
                object_type = object_properties.get("type", "generic")
                material = object_properties.get("material", "unknown")
                
                condition.degradation_factors["base_rate"] = self._get_material_degradation_rate(material)
                condition.degradation_factors["wear_resistance"] = object_properties.get("wear_resistance", 1.0)
                condition.degradation_factors["environmental_resistance"] = object_properties.get("environmental_resistance", 1.0)

        except Exception as e:
            logger.error(f"Error initializing degradation factors: {e}")

    async def _update_degradation_factors(
        self, 
        object_id: str, 
        condition: ObjectCondition, 
        environmental_factors: Dict[str, Any]
    ) -> None:
        """Update degradation factors based on current state."""
        try:
            # Degradation can accelerate as condition worsens
            condition_factor = condition.overall_condition
            
            # Poor condition leads to faster degradation
            if condition_factor < 0.5:
                condition.degradation_factors["condition_acceleration"] = (1.0 - condition_factor) * 0.5
            else:
                condition.degradation_factors.pop("condition_acceleration", None)

        except Exception as e:
            logger.error(f"Error updating degradation factors: {e}")

    async def _check_condition_thresholds(self, object_id: str, condition: ObjectCondition) -> None:
        """Check for condition thresholds and trigger events."""
        try:
            thresholds = self._condition_thresholds.get(object_id, {})
            
            for threshold_name, threshold_value in thresholds.items():
                if condition.overall_condition <= threshold_value:
                    # Trigger threshold event
                    await self._trigger_condition_event(object_id, threshold_name, condition)

        except Exception as e:
            logger.error(f"Error checking condition thresholds: {e}")

    async def _trigger_condition_event(
        self, 
        object_id: str, 
        event_type: str, 
        condition: ObjectCondition
    ) -> None:
        """Trigger condition-related events."""
        try:
            # This would integrate with the event system
            logger.info(f"Condition event triggered for {object_id}: {event_type}")

        except Exception as e:
            logger.error(f"Error triggering condition event: {e}")

    def _calculate_environmental_stress(self, environmental_factors: Dict[str, Any]) -> float:
        """Calculate environmental stress factor."""
        stress = 0.0
        
        temperature = environmental_factors.get("temperature", 20.0)
        humidity = environmental_factors.get("humidity", 50.0)
        
        # Temperature stress
        stress += abs(temperature - 20.0) / 100.0
        
        # Humidity stress
        if humidity > 80.0 or humidity < 20.0:
            stress += abs(humidity - 50.0) / 100.0
        
        return min(1.0, stress)

    def _get_aspect_degradation_multiplier(self, aspect: QualityAspect, usage_data: Dict[str, Any]) -> float:
        """Get degradation multiplier for specific aspect."""
        multipliers = {
            QualityAspect.DURABILITY: 1.0,
            QualityAspect.SHARPNESS: 1.5,  # Sharpness degrades faster
            QualityAspect.EFFICIENCY: 1.2,
            QualityAspect.APPEARANCE: 0.8,  # Appearance degrades slower functionally
            QualityAspect.MAGICAL_POTENCY: 0.5,  # Magic is more stable
            QualityAspect.PURITY: 1.1,
            QualityAspect.STABILITY: 0.9,
        }
        return multipliers.get(aspect, 1.0)

    def _get_aspect_vulnerability(self, aspect: QualityAspect, environment_data: Dict[str, Any]) -> float:
        """Get aspect vulnerability to environmental factors."""
        vulnerabilities = {
            QualityAspect.DURABILITY: 1.0,
            QualityAspect.SHARPNESS: 0.8,
            QualityAspect.EFFICIENCY: 1.2,
            QualityAspect.APPEARANCE: 1.5,  # Appearance most vulnerable to environment
            QualityAspect.MAGICAL_POTENCY: 0.3,
            QualityAspect.PURITY: 1.3,
            QualityAspect.STABILITY: 0.7,
        }
        return vulnerabilities.get(aspect, 1.0)

    def _get_aspect_repair_factor(self, aspect: QualityAspect) -> float:
        """Get repair effectiveness factor for aspect."""
        factors = {
            QualityAspect.DURABILITY: 1.0,
            QualityAspect.SHARPNESS: 0.8,  # Harder to restore sharpness
            QualityAspect.EFFICIENCY: 1.2,  # Easier to restore efficiency
            QualityAspect.APPEARANCE: 1.5,  # Easiest to restore appearance
            QualityAspect.MAGICAL_POTENCY: 0.3,  # Very hard to restore magic
            QualityAspect.PURITY: 0.7,
            QualityAspect.STABILITY: 0.9,
        }
        return factors.get(aspect, 1.0)

    def _get_material_degradation_rate(self, material: str) -> float:
        """Get base degradation rate for material."""
        rates = {
            "iron": 0.002,
            "steel": 0.001,
            "bronze": 0.0015,
            "wood": 0.003,
            "leather": 0.004,
            "cloth": 0.005,
            "stone": 0.0005,
            "magical": 0.0001,
        }
        return rates.get(material, 0.002)

    async def _get_object_properties(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get object properties."""
        try:
            if self.objects:
                return await self.objects.get_object_properties(object_id)
            return {}
        except Exception:
            return {}

    async def _get_repair_requirements(self, object_id: str) -> Dict[str, Any]:
        """Get repair requirements for object."""
        return {"materials": [], "tools": [], "skills": {}}

    async def _validate_repair_materials(
        self, 
        requirements: Dict[str, Any], 
        available_materials: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate repair materials."""
        return True, None

    async def _validate_repair_tools(
        self, 
        requirements: Dict[str, Any], 
        available_tools: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate repair tools."""
        return True, None

    async def _calculate_repair_success_probability(
        self, 
        object_id: str, 
        skill_level: int, 
        materials: List[str], 
        tools: List[str]
    ) -> float:
        """Calculate probability of repair success."""
        base_probability = 0.7
        skill_bonus = min(0.2, skill_level * 0.02)
        return min(1.0, base_probability + skill_bonus)

    async def _calculate_repair_amount(
        self, 
        object_id: str, 
        skill_level: int, 
        materials: List[str], 
        tools: List[str]
    ) -> float:
        """Calculate amount of repair achieved."""
        base_repair = 0.2
        skill_bonus = min(0.3, skill_level * 0.03)
        return min(1.0, base_repair + skill_bonus)

    async def _estimate_repair_materials(self, object_id: str, damage_amount: float) -> List[str]:
        """Estimate materials needed for repair."""
        return ["repair_kit", "metal_scraps"] if damage_amount > 0.5 else ["repair_kit"]

    async def _estimate_repair_skills(self, object_id: str, damage_amount: float) -> Dict[str, int]:
        """Estimate skills needed for repair."""
        skill_level = 3 if damage_amount > 0.5 else 2
        return {"repair": skill_level}

    async def _estimate_repair_tools(self, object_id: str, damage_amount: float) -> List[str]:
        """Estimate tools needed for repair."""
        return ["hammer", "anvil"] if damage_amount > 0.5 else ["hammer"]

    def _initialize_degradation_models(self) -> None:
        """Initialize default degradation models for common object types."""
        
        async def metal_tool_degradation(object_id: str, usage_data: Dict[str, Any], env_data: Dict[str, Any]) -> float:
            """Degradation model for metal tools."""
            base_rate = 0.001
            usage_intensity = usage_data.get("intensity", 0.0)
            return base_rate * (1.0 + usage_intensity)
        
        async def organic_degradation(object_id: str, usage_data: Dict[str, Any], env_data: Dict[str, Any]) -> float:
            """Degradation model for organic materials."""
            base_rate = 0.005
            humidity = env_data.get("humidity", 50.0)
            humidity_factor = 1.0 + (humidity - 50.0) / 100.0
            return base_rate * humidity_factor
        
        self._degradation_models["metal_tool"] = metal_tool_degradation
        self._degradation_models["organic"] = organic_degradation