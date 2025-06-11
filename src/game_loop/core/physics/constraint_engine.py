"""
Physics Constraint Engine for applying realistic physics constraints to actions.

This module provides physics validation, strength calculations, energy expenditure,
and realistic physics simulation for game actions.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from game_loop.core.command_handlers.physical_action_processor import PhysicalActionType

logger = logging.getLogger(__name__)


class PhysicsConstants:
    """Physical constants used in calculations."""
    
    GRAVITY = 9.81  # m/s²
    AIR_DENSITY = 1.225  # kg/m³
    WATER_DENSITY = 1000.0  # kg/m³
    FRICTION_COEFFICIENTS = {
        "wood_on_wood": 0.3,
        "metal_on_metal": 0.15,
        "rubber_on_concrete": 0.7,
        "ice_on_ice": 0.02,
        "default": 0.4,
    }
    DRAG_COEFFICIENTS = {
        "sphere": 0.47,
        "cube": 1.05,
        "cylinder": 0.82,
        "human": 1.0,
        "default": 0.8,
    }


class PhysicsConstraintEngine:
    """
    Apply realistic physics constraints to physical actions.
    
    This class validates physical constraints, calculates requirements,
    and simulates realistic physics behavior for game actions.
    """

    def __init__(self, configuration_manager: Any = None):
        """
        Initialize the physics constraint engine.

        Args:
            configuration_manager: Configuration manager for physics settings
        """
        self.config = configuration_manager
        self._constraint_rules: Dict[str, Callable] = {}
        self._physics_constants: Dict[str, float] = {}
        self._environment_factors: Dict[str, Dict[str, float]] = {}
        self._load_physics_configuration()
        self._initialize_constraint_rules()

    async def validate_physical_constraints(
        self,
        action_type: PhysicalActionType,
        entities: List[str],
        player_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate physical constraints for an action.

        Args:
            action_type: Type of physical action
            entities: List of entities involved in the action
            player_state: Current player state

        Returns:
            Tuple of (constraints_valid, error_message)
        """
        try:
            # Run all relevant constraint rules
            for rule_name, rule_func in self._constraint_rules.items():
                try:
                    is_valid, error_msg = await rule_func(action_type, entities, player_state)
                    if not is_valid:
                        return False, f"{rule_name}: {error_msg}"
                except Exception as e:
                    logger.error(f"Error in constraint rule {rule_name}: {e}")
                    continue

            return True, None

        except Exception as e:
            logger.error(f"Error validating physical constraints: {e}")
            return False, f"Constraint validation error: {str(e)}"

    async def calculate_strength_requirements(
        self,
        action_type: PhysicalActionType,
        target_mass: float,
        difficulty_modifiers: Dict[str, float],
    ) -> float:
        """
        Calculate strength requirements for physical action.

        Args:
            action_type: Type of physical action
            target_mass: Mass of the target object in kg
            difficulty_modifiers: Environmental and situational modifiers

        Returns:
            Required strength value
        """
        try:
            # Base strength requirements by action type
            base_requirements = {
                PhysicalActionType.MOVEMENT: 0.1,  # Minimal strength for movement
                PhysicalActionType.MANIPULATION: 1.0,
                PhysicalActionType.PUSHING: 1.5,
                PhysicalActionType.PULLING: 1.2,
                PhysicalActionType.LIFTING: 2.2,
                PhysicalActionType.CLIMBING: 2.0,
                PhysicalActionType.JUMPING: 1.8,
                PhysicalActionType.OPENING: 0.8,
                PhysicalActionType.CLOSING: 0.7,
                PhysicalActionType.BREAKING: 3.0,
                PhysicalActionType.BUILDING: 2.5,
            }

            base_requirement = base_requirements.get(action_type, 1.0)
            
            # Calculate mass factor
            mass_factor = self._calculate_mass_factor(action_type, target_mass)
            
            # Apply difficulty modifiers
            total_modifier = 1.0
            for modifier_name, modifier_value in difficulty_modifiers.items():
                total_modifier *= modifier_value

            # Calculate leverage factor if tools are involved
            leverage_factor = difficulty_modifiers.get("leverage_factor", 1.0)

            # Final strength requirement
            strength_required = (base_requirement * mass_factor * total_modifier) / leverage_factor

            return max(0.1, strength_required)  # Minimum requirement

        except Exception as e:
            logger.error(f"Error calculating strength requirements: {e}")
            return 1.0  # Default requirement

    async def check_spatial_constraints(
        self,
        action_location: Dict[str, Any],
        required_space: Dict[str, float],
        entities: List[str],
    ) -> bool:
        """
        Check if sufficient space exists for action.

        Args:
            action_location: Location where action is being performed
            required_space: Required space dimensions (width, height, depth)
            entities: Entities involved in the action

        Returns:
            True if sufficient space exists
        """
        try:
            # Get available space at location
            available_space = action_location.get("available_space", {
                "width": 10.0,
                "height": 3.0,
                "depth": 10.0,
            })

            # Check each dimension
            for dimension, required in required_space.items():
                available = available_space.get(dimension, 0.0)
                if available < required:
                    logger.info(f"Insufficient {dimension}: need {required}, have {available}")
                    return False

            # Check for entity overlap
            entity_positions = action_location.get("entity_positions", {})
            for entity in entities:
                entity_space = entity_positions.get(entity, {})
                # Only check overlap if entity has position data
                if entity_space and self._check_space_overlap(entity_space, required_space):
                    logger.info(f"Space overlap detected with entity {entity}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking spatial constraints: {e}")
            return False

    async def validate_structural_integrity(
        self, entity_id: str, applied_force: float, force_type: str
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if entity can withstand applied force.

        Args:
            entity_id: ID of the entity receiving force
            applied_force: Magnitude of applied force in Newtons
            force_type: Type of force (compression, tension, shear, torsion)

        Returns:
            Tuple of (can_withstand, damage_info)
        """
        try:
            # Get entity structural properties
            entity_props = await self._get_entity_structural_properties(entity_id)
            
            # Get maximum force capacity for the force type
            max_force = entity_props.get(f"max_{force_type}_force", 1000.0)
            yield_strength = entity_props.get("yield_strength", 500.0)
            
            damage_info = None

            if applied_force > max_force:
                # Entity breaks
                damage_info = {
                    "type": "structural_failure",
                    "severity": "complete",
                    "description": f"The {entity_id} breaks under the applied force.",
                    "debris": True,
                }
                return False, damage_info

            elif applied_force > yield_strength:
                # Entity is damaged but doesn't break
                damage_severity = min(1.0, (applied_force - yield_strength) / (max_force - yield_strength))
                damage_info = {
                    "type": "structural_damage",
                    "severity": damage_severity,
                    "description": f"The {entity_id} is damaged by the applied force.",
                    "integrity_loss": damage_severity * 0.5,
                }
                return True, damage_info

            return True, None

        except Exception as e:
            logger.error(f"Error validating structural integrity: {e}")
            return True, None  # Default to allowing action

    async def calculate_energy_expenditure(
        self,
        action_type: PhysicalActionType,
        duration: float,
        intensity: float,
        player_stats: Dict[str, Any],
    ) -> float:
        """
        Calculate energy cost for physical action.

        Args:
            action_type: Type of physical action
            duration: Duration of action in seconds
            intensity: Intensity of action (0.0 to 1.0)
            player_stats: Player physical statistics

        Returns:
            Energy cost in energy units
        """
        try:
            # Base metabolic rate (calories per second at rest)
            base_metabolic_rate = 1.2  # ~70 calories per minute
            
            # Activity multipliers for different action types
            activity_multipliers = {
                PhysicalActionType.MOVEMENT: 3.0,
                PhysicalActionType.MANIPULATION: 2.5,
                PhysicalActionType.PUSHING: 4.0,
                PhysicalActionType.PULLING: 3.8,
                PhysicalActionType.LIFTING: 5.5,
                PhysicalActionType.CLIMBING: 6.0,
                PhysicalActionType.JUMPING: 5.0,
                PhysicalActionType.OPENING: 1.5,
                PhysicalActionType.CLOSING: 1.3,
                PhysicalActionType.BREAKING: 7.0,
                PhysicalActionType.BUILDING: 4.5,
            }

            activity_multiplier = activity_multipliers.get(action_type, 2.0)
            
            # Player fitness affects energy efficiency
            fitness_level = player_stats.get("fitness", 50) / 100.0  # 0.0 to 1.0
            efficiency = 0.5 + (fitness_level * 0.5)  # 0.5 to 1.0 efficiency
            
            # Calculate energy expenditure
            base_energy = base_metabolic_rate * activity_multiplier * duration
            intensity_factor = 1.0 + (intensity * 2.0)  # 1.0 to 3.0
            
            total_energy = (base_energy * intensity_factor) / efficiency

            return max(1.0, total_energy)  # Minimum 1 energy unit

        except Exception as e:
            logger.error(f"Error calculating energy expenditure: {e}")
            return 10.0  # Default energy cost

    async def simulate_collision_effects(
        self, entity1: str, entity2: str, impact_force: float
    ) -> Dict[str, Any]:
        """
        Simulate effects of collision between entities.

        Args:
            entity1: First entity in collision
            entity2: Second entity in collision
            impact_force: Force of impact in Newtons

        Returns:
            Dictionary containing collision effects
        """
        try:
            # Get entity properties
            props1 = await self._get_entity_physical_properties(entity1)
            props2 = await self._get_entity_physical_properties(entity2)

            mass1 = props1.get("mass", 10.0)
            mass2 = props2.get("mass", 10.0)
            
            # Calculate momentum transfer (simplified elastic collision)
            total_mass = mass1 + mass2
            velocity_change_1 = (2 * mass2 / total_mass) * (impact_force / mass1)
            velocity_change_2 = (2 * mass1 / total_mass) * (impact_force / mass2)

            # Calculate damage based on impact force and material properties
            damage1 = self._calculate_collision_damage(entity1, impact_force, props1)
            damage2 = self._calculate_collision_damage(entity2, impact_force, props2)

            # Generate sound effects based on materials
            sound_effect = self._generate_collision_sound(props1, props2, impact_force)

            return {
                "entity1_effects": {
                    "velocity_change": velocity_change_1,
                    "damage": damage1,
                    "position_change": velocity_change_1 * 0.1,  # Simplified displacement
                },
                "entity2_effects": {
                    "velocity_change": velocity_change_2,
                    "damage": damage2,
                    "position_change": velocity_change_2 * 0.1,
                },
                "environmental_effects": {
                    "sound": sound_effect,
                    "debris": damage1 > 0.5 or damage2 > 0.5,
                    "impact_crater": impact_force > 1000.0,
                },
                "impact_force": impact_force,
            }

        except Exception as e:
            logger.error(f"Error simulating collision effects: {e}")
            return {"error": str(e)}

    async def check_balance_and_stability(
        self,
        entity_id: str,
        action_type: PhysicalActionType,
        environmental_factors: Dict[str, Any],
    ) -> bool:
        """
        Check if entity maintains balance during action.

        Args:
            entity_id: ID of the entity
            action_type: Type of action being performed
            environmental_factors: Environmental conditions affecting stability

        Returns:
            True if entity maintains balance
        """
        try:
            # Get entity balance properties
            entity_props = await self._get_entity_physical_properties(entity_id)
            center_of_gravity = entity_props.get("center_of_gravity", 0.5)  # 0.0 to 1.0 (low to high)
            stability_factor = entity_props.get("stability", 0.7)
            
            # Action-specific balance challenges
            balance_challenges = {
                PhysicalActionType.MOVEMENT: 0.1,
                PhysicalActionType.CLIMBING: 0.7,
                PhysicalActionType.JUMPING: 0.8,
                PhysicalActionType.PUSHING: 0.4,
                PhysicalActionType.PULLING: 0.3,
                PhysicalActionType.BREAKING: 0.6,
                PhysicalActionType.BUILDING: 0.5,
            }

            challenge_level = balance_challenges.get(action_type, 0.2)
            
            # Environmental factors
            wind_factor = environmental_factors.get("wind_speed", 0.0) / 50.0  # 0.0 to 1.0+
            surface_stability = environmental_factors.get("surface_stability", 1.0)  # 0.0 to 1.0
            visibility = environmental_factors.get("visibility", 1.0)  # 0.0 to 1.0
            
            # Calculate total stability requirement
            total_challenge = challenge_level + wind_factor + (1.0 - surface_stability) + (1.0 - visibility) * 0.2
            
            # Account for center of gravity
            balance_ability = stability_factor * (2.0 - center_of_gravity)
            
            return balance_ability >= total_challenge

        except Exception as e:
            logger.error(f"Error checking balance and stability: {e}")
            return True  # Default to stable

    async def apply_gravity_effects(
        self, entity_id: str, height: float, support_structure: Optional[str]
    ) -> Dict[str, Any]:
        """
        Apply gravity effects to elevated entities.

        Args:
            entity_id: ID of the entity
            height: Height above ground in meters
            support_structure: Optional supporting structure ID

        Returns:
            Dictionary containing gravity effects
        """
        try:
            if height <= 0:
                return {"no_effect": True}

            # Get entity properties
            entity_props = await self._get_entity_physical_properties(entity_id)
            mass = entity_props.get("mass", 10.0)
            
            # Calculate gravitational potential energy
            potential_energy = mass * PhysicsConstants.GRAVITY * height
            
            # Check support structure
            if support_structure:
                support_props = await self._get_entity_structural_properties(support_structure)
                max_load = support_props.get("max_load", 1000.0)
                
                if mass * PhysicsConstants.GRAVITY > max_load:
                    # Support fails
                    fall_effects = await self._calculate_fall_effects(entity_id, height, mass)
                    return {
                        "support_failure": True,
                        "fall_effects": fall_effects,
                        "structure_damage": {
                            "type": "overload_failure",
                            "affected_entity": support_structure,
                        },
                    }

            # Calculate terminal velocity (if falling)
            drag_coefficient = PhysicsConstants.DRAG_COEFFICIENTS["default"]
            cross_sectional_area = entity_props.get("cross_sectional_area", 1.0)
            terminal_velocity = math.sqrt(
                (2 * mass * PhysicsConstants.GRAVITY) / 
                (PhysicsConstants.AIR_DENSITY * drag_coefficient * cross_sectional_area)
            )

            return {
                "potential_energy": potential_energy,
                "terminal_velocity": terminal_velocity,
                "gravitational_force": mass * PhysicsConstants.GRAVITY,
                "stable": support_structure is not None,
            }

        except Exception as e:
            logger.error(f"Error applying gravity effects: {e}")
            return {"error": str(e)}

    def register_constraint_rule(
        self, rule_name: str, validator: Callable, priority: int = 5
    ) -> None:
        """
        Register a new physics constraint rule.

        Args:
            rule_name: Name of the constraint rule
            validator: Validator function for the constraint
            priority: Priority of the rule (higher = earlier execution)
        """
        self._constraint_rules[rule_name] = validator

    async def get_constraint_violations(
        self, action_type: PhysicalActionType, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get list of constraint violations for action.

        Args:
            action_type: Type of physical action
            context: Current action context

        Returns:
            List of constraint violation descriptions
        """
        try:
            violations = []
            entities = context.get("entities", [])
            player_state = context.get("player_state", {})

            # Check all constraint rules
            for rule_name, rule_func in self._constraint_rules.items():
                try:
                    is_valid, error_msg = await rule_func(action_type, entities, player_state)
                    if not is_valid:
                        violations.append({
                            "rule": rule_name,
                            "violation": error_msg,
                            "severity": "high" if "break" in error_msg.lower() else "medium",
                        })
                except Exception as e:
                    logger.error(f"Error checking rule {rule_name}: {e}")

            return violations

        except Exception as e:
            logger.error(f"Error getting constraint violations: {e}")
            return []

    # Private helper methods

    def _load_physics_configuration(self) -> None:
        """Load physics constants and rules from configuration."""
        try:
            # In a full implementation, this would load from configuration files
            self._physics_constants = {
                "gravity": PhysicsConstants.GRAVITY,
                "air_density": PhysicsConstants.AIR_DENSITY,
                "water_density": PhysicsConstants.WATER_DENSITY,
                "default_friction": PhysicsConstants.FRICTION_COEFFICIENTS["default"],
                "default_drag": PhysicsConstants.DRAG_COEFFICIENTS["default"],
            }
            
            # Load environment factors
            self._environment_factors = {
                "normal": {"gravity_multiplier": 1.0, "air_resistance": 1.0},
                "underwater": {"gravity_multiplier": 0.1, "air_resistance": 50.0},
                "space": {"gravity_multiplier": 0.0, "air_resistance": 0.0},
                "low_gravity": {"gravity_multiplier": 0.3, "air_resistance": 1.0},
            }

        except Exception as e:
            logger.error(f"Error loading physics configuration: {e}")

    def _initialize_constraint_rules(self) -> None:
        """Initialize default constraint rules."""
        self._constraint_rules = {
            "mass_limit": self._validate_mass_limits,
            "energy_requirement": self._validate_energy_requirements,
            "spatial_clearance": self._validate_spatial_clearance,
            "structural_integrity": self._validate_structural_integrity_rule,
            "balance_check": self._validate_balance_requirements,
        }

    async def _validate_mass_limits(
        self, action_type: PhysicalActionType, entities: List[str], player_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate mass-based constraints."""
        try:
            player_strength = player_state.get("strength", 50)
            max_lift_capacity = player_strength * 2  # Simplified calculation

            for entity in entities:
                entity_props = await self._get_entity_physical_properties(entity)
                entity_mass = entity_props.get("mass", 10.0)
                
                if action_type in [PhysicalActionType.PUSHING, PhysicalActionType.PULLING]:
                    # Pushing/pulling allows more mass than lifting
                    max_capacity = max_lift_capacity * 3
                else:
                    max_capacity = max_lift_capacity

                if entity_mass > max_capacity:
                    return False, f"Entity {entity} is too heavy ({entity_mass} kg > {max_capacity} kg capacity)"

            return True, ""

        except Exception as e:
            logger.error(f"Error validating mass limits: {e}")
            return True, ""

    async def _validate_energy_requirements(
        self, action_type: PhysicalActionType, entities: List[str], player_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate energy requirements."""
        current_energy = player_state.get("energy", 100)
        min_energy_required = len(entities) * 5  # 5 energy per entity

        if action_type in [PhysicalActionType.CLIMBING, PhysicalActionType.BREAKING]:
            min_energy_required *= 2

        if current_energy < min_energy_required:
            return False, f"Insufficient energy: {min_energy_required} required, {current_energy} available"

        return True, ""

    async def _validate_spatial_clearance(
        self, action_type: PhysicalActionType, entities: List[str], player_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate spatial requirements."""
        # Simplified spatial validation
        current_location = player_state.get("current_location", {})
        available_space = current_location.get("space_rating", 5)  # 1-10 scale

        required_space = 1
        if action_type in [PhysicalActionType.JUMPING, PhysicalActionType.CLIMBING]:
            required_space = 3
        elif action_type in [PhysicalActionType.BUILDING, PhysicalActionType.BREAKING]:
            required_space = 4

        if available_space < required_space:
            return False, f"Insufficient space: {required_space} required, {available_space} available"

        return True, ""

    async def _validate_structural_integrity_rule(
        self, action_type: PhysicalActionType, entities: List[str], player_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate structural integrity constraints."""
        for entity in entities:
            entity_props = await self._get_entity_structural_properties(entity)
            durability = entity_props.get("durability", 100.0)
            
            if durability <= 0:
                return False, f"Entity {entity} is too damaged to interact with"
            
            if action_type == PhysicalActionType.BREAKING and durability > 200:
                return False, f"Entity {entity} is too sturdy to break"

        return True, ""

    async def _validate_balance_requirements(
        self, action_type: PhysicalActionType, entities: List[str], player_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate balance and stability requirements."""
        player_balance = player_state.get("balance_skill", 50)
        
        if action_type in [PhysicalActionType.CLIMBING, PhysicalActionType.JUMPING]:
            if player_balance < 30:
                return False, "Insufficient balance skill for this action"

        return True, ""

    async def _calculate_leverage_factor(
        self, tool_id: Optional[str], action_type: PhysicalActionType
    ) -> float:
        """Calculate leverage factor when using tools."""
        if not tool_id:
            return 1.0

        # Simplified tool effectiveness
        tool_leverages = {
            "crowbar": 3.0,
            "pulley": 2.0,
            "hammer": 1.5,
            "rope": 1.2,
            "lever": 4.0,
        }

        return tool_leverages.get(tool_id, 1.1)

    def _get_environmental_resistance(
        self, location_id: str, action_type: PhysicalActionType
    ) -> float:
        """Get environmental resistance factors."""
        # Simplified environmental resistance
        base_resistance = 1.0
        
        if "underwater" in location_id.lower():
            base_resistance *= 3.0
        elif "muddy" in location_id.lower():
            base_resistance *= 1.5
        elif "icy" in location_id.lower():
            if action_type == PhysicalActionType.MOVEMENT:
                base_resistance *= 0.3  # Slippery
            else:
                base_resistance *= 1.2

        return base_resistance

    def _calculate_mass_factor(self, action_type: PhysicalActionType, mass: float) -> float:
        """Calculate how mass affects action difficulty."""
        # Different actions scale differently with mass
        if action_type == PhysicalActionType.MOVEMENT:
            return 1.0 + (mass / 100.0)  # Mass has minimal effect on self-movement
        elif action_type in [PhysicalActionType.PUSHING, PhysicalActionType.PULLING]:
            return 1.0 + (mass / 50.0)  # Moderate scaling
        elif action_type == PhysicalActionType.LIFTING:
            return 1.0 + (mass / 25.0)  # High scaling for lifting
        elif action_type in [PhysicalActionType.CLIMBING, PhysicalActionType.JUMPING]:
            return 1.0 + (mass / 30.0)  # Higher scaling for acrobatic actions
        else:
            return 1.0 + (mass / 75.0)  # Default scaling

    def _check_space_overlap(self, space1: Dict[str, float], space2: Dict[str, float]) -> bool:
        """Check if two spaces overlap."""
        # Simplified 2D overlap check
        x1_min, x1_max = space1.get("x", 0), space1.get("x", 0) + space1.get("width", 1)
        y1_min, y1_max = space1.get("y", 0), space1.get("y", 0) + space1.get("height", 1)
        
        x2_min, x2_max = space2.get("x", 0), space2.get("x", 0) + space2.get("width", 1)
        y2_min, y2_max = space2.get("y", 0), space2.get("y", 0) + space2.get("height", 1)
        
        return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)

    async def _get_entity_structural_properties(self, entity_id: str) -> Dict[str, Any]:
        """Get structural properties of an entity."""
        # Simulate entity structural data
        return {
            "max_compression_force": 1000.0,
            "max_tension_force": 800.0,
            "max_shear_force": 600.0,
            "yield_strength": 500.0,
            "durability": 100.0,
            "max_load": 500.0,
        }

    async def _get_entity_physical_properties(self, entity_id: str) -> Dict[str, Any]:
        """Get physical properties of an entity."""
        # Simulate entity physical data
        return {
            "mass": 10.0,
            "center_of_gravity": 0.5,
            "stability": 0.7,
            "cross_sectional_area": 1.0,
            "material": "wood",
        }

    def _calculate_collision_damage(
        self, entity_id: str, impact_force: float, entity_props: Dict[str, Any]
    ) -> float:
        """Calculate damage from collision."""
        durability = entity_props.get("durability", 100.0)
        damage_threshold = entity_props.get("damage_threshold", 100.0)
        
        if impact_force > damage_threshold:
            damage_ratio = min(1.0, (impact_force - damage_threshold) / damage_threshold)
            return damage_ratio
        
        return 0.0

    def _generate_collision_sound(
        self, props1: Dict[str, Any], props2: Dict[str, Any], impact_force: float
    ) -> str:
        """Generate collision sound effect description."""
        material1 = props1.get("material", "unknown")
        material2 = props2.get("material", "unknown")
        
        if impact_force > 500:
            intensity = "loud"
        elif impact_force > 100:
            intensity = "moderate"
        else:
            intensity = "soft"
        
        return f"A {intensity} {material1}-on-{material2} collision sound"

    async def _calculate_fall_effects(
        self, entity_id: str, height: float, mass: float
    ) -> Dict[str, Any]:
        """Calculate effects of falling."""
        # Simplified fall damage calculation
        kinetic_energy = mass * PhysicsConstants.GRAVITY * height
        damage = min(1.0, kinetic_energy / 1000.0)  # Normalize to 0-1
        
        return {
            "impact_energy": kinetic_energy,
            "damage": damage,
            "sound_effect": "crash" if damage > 0.5 else "thud",
            "debris": damage > 0.7,
        }