"""
Object Interaction System for complex object interactions and tool usage.

This module provides sophisticated object interaction mechanics including tool usage,
object state transitions, skill-based success probability, and interaction chaining.
"""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ObjectInteractionType(Enum):
    """Types of interactions that can be performed with objects."""
    
    EXAMINE = "examine"
    USE = "use"
    COMBINE = "combine"
    TRANSFORM = "transform"
    DISASSEMBLE = "disassemble"
    REPAIR = "repair"
    ENHANCE = "enhance"
    CONSUME = "consume"


@dataclass
class InteractionResult:
    """Result of an object interaction."""
    
    success: bool
    interaction_type: ObjectInteractionType
    source_object: str
    target_object: Optional[str]
    tool_used: Optional[str]
    state_changes: Dict[str, Any]
    products: List[str]
    byproducts: List[str]
    energy_cost: float
    time_elapsed: float
    skill_experience: Dict[str, float]
    description: str
    error_message: Optional[str] = None


class ObjectInteractionSystem:
    """
    Manage complex object interactions with tools, skills, and state transitions.
    
    This class provides comprehensive object interaction mechanics including:
    - Tool usage with compatibility checking
    - Skill-based success probability
    - Object state transitions and conditions
    - Recipe-based combinations and transformations
    - Interaction chaining and dependencies
    """

    def __init__(
        self, 
        object_manager: Any = None, 
        physics_engine: Any = None, 
        skill_manager: Any = None, 
        recipe_manager: Any = None
    ):
        """
        Initialize the object interaction system.

        Args:
            object_manager: Manager for object data and properties
            physics_engine: Physics engine for interaction constraints
            skill_manager: Manager for skill levels and experience
            recipe_manager: Manager for combination recipes
        """
        self.objects = object_manager
        self.physics = physics_engine
        self.skills = skill_manager
        self.recipes = recipe_manager
        self._interaction_handlers: Dict[ObjectInteractionType, List[Callable]] = {}
        self._compatibility_matrix: Dict[str, Dict[str, float]] = {}
        self._state_machines: Dict[str, Dict[str, Any]] = {}
        self._interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        self._success_modifiers: Dict[str, Callable] = {}
        self._initialize_interaction_handlers()

    async def process_object_interaction(
        self,
        interaction_type: ObjectInteractionType,
        source_object: str,
        target_object: Optional[str],
        tool_object: Optional[str],
        context: Dict[str, Any]
    ) -> InteractionResult:
        """
        Process a complex object interaction.

        Args:
            interaction_type: Type of interaction to perform
            source_object: Primary object being used/manipulated
            target_object: Secondary object (for combinations, etc.)
            tool_object: Tool being used for the interaction
            context: Current game context including player state

        Returns:
            InteractionResult with detailed outcome information
        """
        try:
            interaction_id = str(uuid.uuid4())
            start_time = asyncio.get_event_loop().time()
            
            logger.info(f"Processing {interaction_type.value} interaction: {source_object} -> {target_object} with {tool_object}")
            
            # Validate interaction requirements
            is_valid, requirements_errors = await self.validate_interaction_requirements(
                interaction_type, [source_object] + ([target_object] if target_object else []), 
                context.get("player_state", {})
            )
            
            if not is_valid:
                return InteractionResult(
                    success=False,
                    interaction_type=interaction_type,
                    source_object=source_object,
                    target_object=target_object,
                    tool_used=tool_object,
                    state_changes={},
                    products=[],
                    byproducts=[],
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    skill_experience={},
                    description="Interaction failed validation.",
                    error_message="; ".join(requirements_errors),
                )
            
            # Calculate success probability
            success_probability = await self.calculate_interaction_success_probability(
                interaction_type, [source_object] + ([target_object] if target_object else []),
                context.get("player_state", {}).get("skills", {})
            )
            
            # Apply modifiers
            for modifier_name, modifier_func in self._success_modifiers.items():
                success_probability = await modifier_func(
                    success_probability, interaction_type, source_object, target_object, tool_object, context
                )
            
            # Determine success
            import random
            is_successful = random.random() < success_probability
            
            # Get interaction handler
            handlers = self._interaction_handlers.get(interaction_type, [])
            if not handlers:
                handlers = [self._default_interaction_handler]
            
            # Execute interaction
            result_data = {}
            for handler_info in handlers:
                # Extract handler function from tuple (priority, handler)
                if isinstance(handler_info, tuple):
                    _, handler_func = handler_info
                else:
                    handler_func = handler_info
                
                handler_result = await handler_func(
                    source_object, target_object, tool_object, context, is_successful
                )
                result_data.update(handler_result)
            
            # Calculate costs and experience
            energy_cost = await self._calculate_energy_cost(interaction_type, source_object, target_object, tool_object)
            time_elapsed = asyncio.get_event_loop().time() - start_time
            skill_experience = await self._calculate_skill_experience(
                interaction_type, is_successful, context.get("player_state", {})
            )
            
            # Apply wear and degradation
            if is_successful and tool_object:
                await self.apply_wear_and_degradation(tool_object, 1.0, interaction_type)
            
            # Create result
            result = InteractionResult(
                success=is_successful,
                interaction_type=interaction_type,
                source_object=source_object,
                target_object=target_object,
                tool_used=tool_object,
                state_changes=result_data.get("state_changes", {}),
                products=result_data.get("products", []),
                byproducts=result_data.get("byproducts", []),
                energy_cost=energy_cost,
                time_elapsed=time_elapsed,
                skill_experience=skill_experience,
                description=result_data.get("description", f"Performed {interaction_type.value} interaction."),
                error_message=result_data.get("error_message") if not is_successful else None,
            )
            
            # Record interaction
            await self._record_interaction(interaction_id, result, context)
            
            return result

        except Exception as e:
            logger.error(f"Error processing object interaction: {e}")
            return InteractionResult(
                success=False,
                interaction_type=interaction_type,
                source_object=source_object,
                target_object=target_object,
                tool_used=tool_object,
                state_changes={},
                products=[],
                byproducts=[],
                energy_cost=0.0,
                time_elapsed=0.0,
                skill_experience={},
                description="Interaction failed due to system error.",
                error_message=str(e),
            )

    async def validate_interaction_requirements(
        self,
        interaction_type: ObjectInteractionType,
        objects: List[str],
        player_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate all requirements for an object interaction.

        Args:
            interaction_type: Type of interaction
            objects: List of objects involved
            player_state: Current player state

        Returns:
            Tuple of (is_valid, list_of_requirement_errors)
        """
        try:
            errors = []
            
            # Check object existence and accessibility
            for obj_id in objects:
                if not await self._validate_object_exists(obj_id):
                    errors.append(f"Object {obj_id} does not exist")
                elif not await self._validate_object_accessible(obj_id, player_state):
                    errors.append(f"Object {obj_id} is not accessible")
            
            # Check interaction-specific requirements
            if interaction_type == ObjectInteractionType.COMBINE:
                if len(objects) < 2:
                    errors.append("Combine interaction requires at least 2 objects")
            
            elif interaction_type == ObjectInteractionType.REPAIR:
                # Check if object is damaged and repairable
                if objects:
                    obj_condition = await self._get_object_condition(objects[0])
                    if obj_condition >= 1.0:
                        errors.append(f"Object {objects[0]} does not need repair")
            
            elif interaction_type == ObjectInteractionType.CONSUME:
                # Check if object is consumable
                if objects:
                    obj_properties = await self._get_object_properties(objects[0])
                    if not obj_properties.get("consumable", False):
                        errors.append(f"Object {objects[0]} is not consumable")
            
            # Check skill requirements
            skill_requirements = await self._get_interaction_skill_requirements(interaction_type, objects)
            player_skills = player_state.get("skills", {})
            
            for skill, required_level in skill_requirements.items():
                player_level = player_skills.get(skill, 0)
                if player_level < required_level:
                    errors.append(f"Requires {skill} level {required_level}, have {player_level}")
            
            # Check energy requirements
            min_energy = await self._calculate_energy_cost(interaction_type, *objects[:3])
            current_energy = player_state.get("energy", 100)
            if current_energy < min_energy:
                errors.append(f"Insufficient energy: need {min_energy}, have {current_energy}")
            
            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating interaction requirements: {e}")
            return False, [str(e)]

    async def get_available_interactions(
        self, 
        object_id: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get all available interactions for an object in current context.

        Args:
            object_id: Object to get interactions for
            context: Current game context

        Returns:
            List of available interaction options
        """
        try:
            interactions = []
            obj_properties = await self._get_object_properties(object_id)
            
            if not obj_properties:
                return interactions
            
            # Basic interactions available for all objects
            interactions.append({
                "type": ObjectInteractionType.EXAMINE,
                "description": f"Examine the {obj_properties.get('name', object_id)}",
                "requirements": [],
                "success_probability": 1.0,
            })
            
            # Context-specific interactions
            if obj_properties.get("usable", False):
                interactions.append({
                    "type": ObjectInteractionType.USE,
                    "description": f"Use the {obj_properties.get('name', object_id)}",
                    "requirements": [],
                    "success_probability": 0.9,
                })
            
            if obj_properties.get("consumable", False):
                interactions.append({
                    "type": ObjectInteractionType.CONSUME,
                    "description": f"Consume the {obj_properties.get('name', object_id)}",
                    "requirements": [],
                    "success_probability": 1.0,
                })
            
            # Condition-based interactions
            obj_condition = await self._get_object_condition(object_id)
            if obj_condition < 1.0:
                interactions.append({
                    "type": ObjectInteractionType.REPAIR,
                    "description": f"Repair the {obj_properties.get('name', object_id)}",
                    "requirements": ["repair skill", "repair materials"],
                    "success_probability": 0.7,
                })
            
            # Tool-based interactions
            if obj_properties.get("disassemblable", False):
                interactions.append({
                    "type": ObjectInteractionType.DISASSEMBLE,
                    "description": f"Disassemble the {obj_properties.get('name', object_id)}",
                    "requirements": ["tools", "disassembly skill"],
                    "success_probability": 0.6,
                })
            
            return interactions

        except Exception as e:
            logger.error(f"Error getting available interactions: {e}")
            return []

    async def execute_tool_interaction(
        self, 
        tool_id: str, 
        target_id: str, 
        action: str,
        context: Dict[str, Any]
    ) -> InteractionResult:
        """
        Execute interaction using a tool on a target object.

        Args:
            tool_id: Tool being used
            target_id: Target object
            action: Action to perform
            context: Game context

        Returns:
            InteractionResult with tool usage outcome
        """
        try:
            # Map action to interaction type
            interaction_type_map = {
                "cut": ObjectInteractionType.TRANSFORM,
                "repair": ObjectInteractionType.REPAIR,
                "enhance": ObjectInteractionType.ENHANCE,
                "disassemble": ObjectInteractionType.DISASSEMBLE,
            }
            
            interaction_type = interaction_type_map.get(action, ObjectInteractionType.USE)
            
            # Check tool compatibility
            tool_properties = await self._get_object_properties(tool_id)
            target_properties = await self._get_object_properties(target_id)
            
            if not tool_properties or not target_properties:
                return InteractionResult(
                    success=False,
                    interaction_type=interaction_type,
                    source_object=target_id,
                    target_object=None,
                    tool_used=tool_id,
                    state_changes={},
                    products=[],
                    byproducts=[],
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    skill_experience={},
                    description="Tool or target not found.",
                    error_message="Invalid tool or target object",
                )
            
            # Check compatibility
            tool_compatible_actions = tool_properties.get("compatible_actions", [])
            if action not in tool_compatible_actions:
                return InteractionResult(
                    success=False,
                    interaction_type=interaction_type,
                    source_object=target_id,
                    target_object=None,
                    tool_used=tool_id,
                    state_changes={},
                    products=[],
                    byproducts=[],
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    skill_experience={},
                    description=f"Tool {tool_id} cannot {action} {target_id}.",
                    error_message=f"Tool incompatible with action {action}",
                )
            
            # Process the interaction
            return await self.process_object_interaction(
                interaction_type, target_id, None, tool_id, context
            )

        except Exception as e:
            logger.error(f"Error executing tool interaction: {e}")
            return InteractionResult(
                success=False,
                interaction_type=ObjectInteractionType.USE,
                source_object=target_id,
                target_object=None,
                tool_used=tool_id,
                state_changes={},
                products=[],
                byproducts=[],
                energy_cost=0.0,
                time_elapsed=0.0,
                skill_experience={},
                description="Tool interaction failed.",
                error_message=str(e),
            )

    async def process_object_combination(
        self, 
        primary_object: str, 
        secondary_objects: List[str],
        recipe_id: Optional[str], 
        context: Dict[str, Any]
    ) -> InteractionResult:
        """
        Combine multiple objects according to recipe or discovery.

        Args:
            primary_object: Main object in combination
            secondary_objects: Additional objects to combine
            recipe_id: Specific recipe to use (optional)
            context: Game context

        Returns:
            InteractionResult with combination outcome
        """
        try:
            all_objects = [primary_object] + secondary_objects
            
            # If recipe specified, validate it
            if recipe_id:
                recipe = await self._get_recipe(recipe_id)
                if not recipe:
                    return self._failed_interaction_result(
                        ObjectInteractionType.COMBINE, primary_object, None, None,
                        f"Recipe {recipe_id} not found"
                    )
                
                # Validate recipe requirements
                recipe_valid, recipe_errors = await self._validate_recipe_requirements(recipe, all_objects)
                if not recipe_valid:
                    return self._failed_interaction_result(
                        ObjectInteractionType.COMBINE, primary_object, None, None,
                        f"Recipe validation failed: {'; '.join(recipe_errors)}"
                    )
            
            # Process combination
            return await self.process_object_interaction(
                ObjectInteractionType.COMBINE, primary_object, 
                secondary_objects[0] if secondary_objects else None, None, context
            )

        except Exception as e:
            logger.error(f"Error processing object combination: {e}")
            return self._failed_interaction_result(
                ObjectInteractionType.COMBINE, primary_object, None, None, str(e)
            )

    async def handle_object_state_transition(
        self, 
        object_id: str, 
        trigger_event: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle state transitions for dynamic objects.

        Args:
            object_id: Object undergoing state transition
            trigger_event: Event triggering the transition
            context: Game context

        Returns:
            Dict with transition results
        """
        try:
            if object_id not in self._state_machines:
                # Create default state machine
                self._state_machines[object_id] = {
                    "current_state": "default",
                    "states": {"default": {"transitions": {}}},
                    "transition_history": [],
                }
            
            state_machine = self._state_machines[object_id]
            current_state = state_machine["current_state"]
            
            # Check if transition is valid
            valid_transitions = state_machine["states"].get(current_state, {}).get("transitions", {})
            
            if trigger_event in valid_transitions:
                new_state = valid_transitions[trigger_event]
                
                # Execute transition
                transition_result = await self._execute_state_transition(
                    object_id, current_state, new_state, trigger_event, context
                )
                
                # Update state machine
                state_machine["current_state"] = new_state
                state_machine["transition_history"].append({
                    "from_state": current_state,
                    "to_state": new_state,
                    "trigger": trigger_event,
                    "timestamp": asyncio.get_event_loop().time(),
                    "result": transition_result,
                })
                
                return {
                    "success": True,
                    "from_state": current_state,
                    "to_state": new_state,
                    "trigger": trigger_event,
                    "effects": transition_result,
                }
            else:
                return {
                    "success": False,
                    "error": f"Invalid transition {trigger_event} from state {current_state}",
                    "valid_transitions": list(valid_transitions.keys()),
                }

        except Exception as e:
            logger.error(f"Error handling object state transition: {e}")
            return {"success": False, "error": str(e)}

    async def calculate_interaction_success_probability(
        self, 
        interaction_type: ObjectInteractionType,
        objects: List[str], 
        player_skills: Dict[str, int]
    ) -> float:
        """
        Calculate probability of interaction success.

        Args:
            interaction_type: Type of interaction
            objects: Objects involved in interaction
            player_skills: Player's current skill levels

        Returns:
            Probability of success (0.0 to 1.0)
        """
        try:
            # Base success probabilities
            base_probabilities = {
                ObjectInteractionType.EXAMINE: 1.0,
                ObjectInteractionType.USE: 0.9,
                ObjectInteractionType.COMBINE: 0.7,
                ObjectInteractionType.TRANSFORM: 0.6,
                ObjectInteractionType.DISASSEMBLE: 0.5,
                ObjectInteractionType.REPAIR: 0.4,
                ObjectInteractionType.ENHANCE: 0.3,
                ObjectInteractionType.CONSUME: 1.0,
            }
            
            base_probability = base_probabilities.get(interaction_type, 0.5)
            
            # Apply skill modifiers
            relevant_skills = await self._get_interaction_skill_requirements(interaction_type, objects)
            skill_modifier = 1.0
            
            for skill, required_level in relevant_skills.items():
                player_level = player_skills.get(skill, 0)
                if player_level >= required_level:
                    # Bonus for higher skill
                    skill_modifier += (player_level - required_level) * 0.05
                else:
                    # Penalty for lower skill
                    skill_modifier -= (required_level - player_level) * 0.1
            
            # Apply object condition modifiers
            condition_modifier = 1.0
            for obj_id in objects:
                obj_condition = await self._get_object_condition(obj_id)
                condition_modifier *= (0.5 + 0.5 * obj_condition)  # Condition affects success
            
            # Calculate final probability
            final_probability = base_probability * skill_modifier * condition_modifier
            
            # Clamp to valid range
            return max(0.0, min(1.0, final_probability))

        except Exception as e:
            logger.error(f"Error calculating interaction success probability: {e}")
            return 0.5

    async def apply_wear_and_degradation(
        self, 
        object_id: str, 
        usage_intensity: float,
        interaction_type: ObjectInteractionType
    ) -> Dict[str, Any]:
        """
        Apply wear and degradation to objects from use.

        Args:
            object_id: Object to apply wear to
            usage_intensity: Intensity of usage (0.0 to 1.0+)
            interaction_type: Type of interaction causing wear

        Returns:
            Dict with wear application results
        """
        try:
            obj_properties = await self._get_object_properties(object_id)
            if not obj_properties:
                return {"error": f"Object {object_id} not found"}
            
            # Calculate wear amount based on interaction type
            wear_rates = {
                ObjectInteractionType.USE: 0.01,
                ObjectInteractionType.COMBINE: 0.005,
                ObjectInteractionType.TRANSFORM: 0.02,
                ObjectInteractionType.DISASSEMBLE: 0.03,
                ObjectInteractionType.REPAIR: 0.005,
                ObjectInteractionType.ENHANCE: 0.01,
            }
            
            base_wear = wear_rates.get(interaction_type, 0.01)
            actual_wear = base_wear * usage_intensity
            
            # Apply durability modifier
            durability = obj_properties.get("durability", 1.0)
            wear_resistance = obj_properties.get("wear_resistance", 1.0)
            actual_wear = actual_wear / (durability * wear_resistance)
            
            # Get current condition
            current_condition = await self._get_object_condition(object_id)
            new_condition = max(0.0, current_condition - actual_wear)
            
            # Update object condition
            await self._set_object_condition(object_id, new_condition)
            
            return {
                "object_id": object_id,
                "interaction_type": interaction_type.value,
                "wear_applied": actual_wear,
                "old_condition": current_condition,
                "new_condition": new_condition,
                "broken": new_condition <= 0.0,
            }

        except Exception as e:
            logger.error(f"Error applying wear and degradation: {e}")
            return {"error": str(e)}

    def register_interaction_handler(
        self, 
        interaction_type: ObjectInteractionType,
        handler: Callable, 
        priority: int = 5
    ) -> None:
        """
        Register custom interaction handler.

        Args:
            interaction_type: Type of interaction to handle
            handler: Handler function
            priority: Handler priority (higher = earlier execution)
        """
        if interaction_type not in self._interaction_handlers:
            self._interaction_handlers[interaction_type] = []
        
        self._interaction_handlers[interaction_type].append((priority, handler))
        # Sort by priority
        self._interaction_handlers[interaction_type].sort(key=lambda x: x[0], reverse=True)

    async def discover_new_interactions(
        self, 
        objects: List[str], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Discover previously unknown interaction possibilities.

        Args:
            objects: Objects to analyze for interactions
            context: Game context

        Returns:
            List of discovered interaction possibilities
        """
        try:
            discoveries = []
            
            # Check all combinations of objects
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i + 1:], i + 1):
                    compatibility = await self._check_object_compatibility(obj1, obj2)
                    if compatibility > 0.7:  # High compatibility threshold
                        discoveries.append({
                            "objects": [obj1, obj2],
                            "interaction_type": ObjectInteractionType.COMBINE,
                            "compatibility": compatibility,
                            "discovery_method": "compatibility_analysis",
                        })
            
            return discoveries

        except Exception as e:
            logger.error(f"Error discovering new interactions: {e}")
            return []

    # Private helper methods

    async def _validate_object_exists(self, object_id: str) -> bool:
        """Check if object exists."""
        try:
            if self.objects:
                return await self.objects.object_exists(object_id)
            return True  # Assume exists for testing
        except Exception:
            return False

    async def _validate_object_accessible(self, object_id: str, player_state: Dict[str, Any]) -> bool:
        """Check if object is accessible to player."""
        # In a full implementation, this would check:
        # - Object location vs player location
        # - Object accessibility (locked, hidden, etc.)
        # - Player permissions
        return True

    async def _get_object_properties(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get object properties."""
        try:
            if self.objects:
                return await self.objects.get_object_properties(object_id)
            else:
                # Fallback for testing
                return {
                    "name": object_id.replace("_", " ").title(),
                    "usable": True,
                    "consumable": "food" in object_id.lower(),
                    "disassemblable": "machine" in object_id.lower(),
                    "durability": 1.0,
                    "wear_resistance": 1.0,
                    "compatible_actions": ["use", "examine"],
                }
        except Exception:
            return None

    async def _get_object_condition(self, object_id: str) -> float:
        """Get object condition (0.0 to 1.0)."""
        # This would integrate with the condition manager
        return 1.0  # Default to perfect condition

    async def _set_object_condition(self, object_id: str, condition: float) -> None:
        """Set object condition."""
        # This would integrate with the condition manager
        pass

    async def _get_interaction_skill_requirements(
        self, 
        interaction_type: ObjectInteractionType, 
        objects: List[str]
    ) -> Dict[str, int]:
        """Get skill requirements for interaction."""
        skill_requirements = {
            ObjectInteractionType.REPAIR: {"repair": 3, "tool_use": 2},
            ObjectInteractionType.DISASSEMBLE: {"disassembly": 4, "tool_use": 3},
            ObjectInteractionType.ENHANCE: {"crafting": 5, "enhancement": 4},
            ObjectInteractionType.COMBINE: {"crafting": 2},
        }
        return skill_requirements.get(interaction_type, {})

    async def _calculate_energy_cost(
        self, 
        interaction_type: ObjectInteractionType, 
        source_object: str, 
        target_object: Optional[str] = None, 
        tool_object: Optional[str] = None
    ) -> float:
        """Calculate energy cost for interaction."""
        base_costs = {
            ObjectInteractionType.EXAMINE: 1.0,
            ObjectInteractionType.USE: 5.0,
            ObjectInteractionType.COMBINE: 10.0,
            ObjectInteractionType.TRANSFORM: 15.0,
            ObjectInteractionType.DISASSEMBLE: 20.0,
            ObjectInteractionType.REPAIR: 25.0,
            ObjectInteractionType.ENHANCE: 30.0,
            ObjectInteractionType.CONSUME: 2.0,
        }
        return base_costs.get(interaction_type, 10.0)

    async def _calculate_skill_experience(
        self, 
        interaction_type: ObjectInteractionType, 
        success: bool, 
        player_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate skill experience gained."""
        if not success:
            return {}
        
        base_experience = {
            ObjectInteractionType.USE: {"tool_use": 1.0},
            ObjectInteractionType.COMBINE: {"crafting": 2.0},
            ObjectInteractionType.REPAIR: {"repair": 3.0},
            ObjectInteractionType.DISASSEMBLE: {"disassembly": 2.5},
            ObjectInteractionType.ENHANCE: {"enhancement": 4.0},
        }
        
        return base_experience.get(interaction_type, {})

    async def _check_object_compatibility(self, object1: str, object2: str) -> float:
        """Check compatibility between two objects."""
        # This would use the compatibility matrix
        cache_key = f"{object1}:{object2}"
        reverse_key = f"{object2}:{object1}"
        
        if cache_key in self._compatibility_matrix:
            return self._compatibility_matrix[cache_key].get("compatibility", 0.0)
        elif reverse_key in self._compatibility_matrix:
            return self._compatibility_matrix[reverse_key].get("compatibility", 0.0)
        
        # Default compatibility based on object types
        return 0.5

    async def _execute_state_transition(
        self, 
        object_id: str, 
        from_state: str, 
        to_state: str, 
        trigger: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a state transition."""
        return {
            "state_changed": True,
            "effects": [],
            "duration": 1.0,
        }

    async def _record_interaction(
        self, 
        interaction_id: str, 
        result: InteractionResult, 
        context: Dict[str, Any]
    ) -> None:
        """Record interaction in history."""
        player_id = context.get("player_id", "unknown")
        
        if player_id not in self._interaction_history:
            self._interaction_history[player_id] = []
        
        self._interaction_history[player_id].append({
            "interaction_id": interaction_id,
            "timestamp": asyncio.get_event_loop().time(),
            "result": result,
            "context": context,
        })
        
        # Limit history size
        max_history = 1000
        if len(self._interaction_history[player_id]) > max_history:
            self._interaction_history[player_id] = self._interaction_history[player_id][-max_history:]

    def _failed_interaction_result(
        self, 
        interaction_type: ObjectInteractionType, 
        source_object: str, 
        target_object: Optional[str], 
        tool_object: Optional[str], 
        error_message: str
    ) -> InteractionResult:
        """Create a failed interaction result."""
        return InteractionResult(
            success=False,
            interaction_type=interaction_type,
            source_object=source_object,
            target_object=target_object,
            tool_used=tool_object,
            state_changes={},
            products=[],
            byproducts=[],
            energy_cost=0.0,
            time_elapsed=0.0,
            skill_experience={},
            description="Interaction failed.",
            error_message=error_message,
        )

    async def _get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get recipe by ID."""
        if self.recipes:
            return await self.recipes.get_recipe(recipe_id)
        return None

    async def _validate_recipe_requirements(
        self, 
        recipe: Dict[str, Any], 
        available_objects: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate recipe requirements against available objects."""
        # This would check recipe components, tools, etc.
        return True, []

    def _initialize_interaction_handlers(self) -> None:
        """Initialize default interaction handlers."""
        
        async def examine_handler(source: str, target: Optional[str], tool: Optional[str], 
                                context: Dict[str, Any], success: bool) -> Dict[str, Any]:
            """Default examine handler."""
            obj_props = await self._get_object_properties(source)
            description = f"You examine the {obj_props.get('name', source)}."
            if obj_props and obj_props.get("description"):
                description += f" {obj_props['description']}"
            
            return {
                "description": description,
                "state_changes": {},
                "products": [],
                "byproducts": [],
            }
        
        async def use_handler(source: str, target: Optional[str], tool: Optional[str], 
                            context: Dict[str, Any], success: bool) -> Dict[str, Any]:
            """Default use handler."""
            if success:
                return {
                    "description": f"You successfully use the {source}.",
                    "state_changes": {f"{source}_used": True},
                    "products": [],
                    "byproducts": [],
                }
            else:
                return {
                    "description": f"You fail to use the {source} properly.",
                    "state_changes": {},
                    "products": [],
                    "byproducts": [],
                    "error_message": "Usage failed",
                }
        
        self.register_interaction_handler(ObjectInteractionType.EXAMINE, examine_handler)
        self.register_interaction_handler(ObjectInteractionType.USE, use_handler)
        
        # Store just the handler function for default handler
        self._default_interaction_handler = use_handler