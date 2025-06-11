"""
Crafting Manager for complex item creation and assembly mechanics.

This module provides comprehensive crafting system with recipe management,
skill-based success probability, component tracking, and crafting station requirements.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CraftingComplexity(Enum):
    """Complexity levels for crafting recipes."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MASTER = "master"
    LEGENDARY = "legendary"


@dataclass
class CraftingRecipe:
    """Complete specification for a crafting recipe."""

    recipe_id: str
    name: str
    description: str
    required_components: dict[str, int]
    optional_components: dict[str, int]
    required_tools: list[str]
    required_skills: dict[str, int]
    crafting_stations: list[str]
    complexity: CraftingComplexity
    base_success_chance: float
    crafting_time: float
    energy_cost: float
    products: dict[str, int]
    byproducts: dict[str, int]
    skill_experience: dict[str, float]


class CraftingManager:
    """
    Manage complex crafting, assembly, and transformation mechanics.

    This class provides comprehensive crafting management including:
    - Recipe management and discovery
    - Component requirement validation
    - Skill-based success probability calculation
    - Dynamic recipe generation
    - Crafting station requirements and modifiers
    - Multi-step crafting processes
    """

    def __init__(
        self,
        object_manager: Any = None,
        inventory_manager: Any = None,
        skill_manager: Any = None,
        physics_engine: Any = None,
    ):
        """
        Initialize the crafting manager.

        Args:
            object_manager: Manager for object data and properties
            inventory_manager: Manager for inventory operations
            skill_manager: Manager for skill levels and experience
            physics_engine: Physics engine for crafting constraints
        """
        self.objects = object_manager
        self.inventory = inventory_manager
        self.skills = skill_manager
        self.physics = physics_engine
        self._recipe_registry: dict[str, CraftingRecipe] = {}
        self._crafting_stations: dict[str, dict[str, Any]] = {}
        self._active_crafting_sessions: dict[str, dict[str, Any]] = {}
        self._crafting_modifiers: dict[str, Callable] = {}
        self._discovery_patterns: dict[str, list[dict[str, Any]]] = {}
        self._crafting_history: dict[str, list[dict[str, Any]]] = {}
        self._initialize_recipes()

    async def start_crafting_session(
        self, crafter_id: str, recipe_id: str, component_sources: dict[str, str]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Start a new crafting session with specified components.

        Args:
            crafter_id: ID of the entity doing the crafting
            recipe_id: Recipe to execute
            component_sources: Mapping of components to their source inventories

        Returns:
            Tuple of (success, session_data)
        """
        try:
            if recipe_id not in self._recipe_registry:
                return False, {"error": f"Recipe {recipe_id} not found"}

            recipe = self._recipe_registry[recipe_id]
            session_id = f"craft_{crafter_id}_{uuid.uuid4().hex[:8]}"

            # Validate basic crafting requirements (skills, tools, stations)
            is_valid, validation_errors = await self._validate_basic_requirements(
                recipe, crafter_id
            )

            if not is_valid:
                return False, {
                    "error": "Validation failed",
                    "details": validation_errors,
                }

            # Reserve components from inventories
            reserved_components = {}
            for component, quantity in recipe.required_components.items():
                source_inventory = component_sources.get(component)
                if not source_inventory:
                    return False, {
                        "error": f"No source specified for component {component}"
                    }

                # Check availability and reserve
                if self.inventory:
                    success, result = await self.inventory.remove_item(
                        source_inventory, component, quantity
                    )
                    if not success:
                        # Rollback previous reservations
                        await self._rollback_component_reservations(reserved_components)
                        return False, {
                            "error": f"Failed to reserve {component}: {result.get('error')}"
                        }

                    reserved_components[component] = {
                        "quantity": quantity,
                        "source": source_inventory,
                        "reserved_at": asyncio.get_event_loop().time(),
                    }

            # Create crafting session
            session = {
                "session_id": session_id,
                "crafter_id": crafter_id,
                "recipe": recipe,
                "reserved_components": reserved_components,
                "start_time": asyncio.get_event_loop().time(),
                "current_step": 0,
                "total_steps": self._calculate_crafting_steps(recipe),
                "progress": 0.0,
                "status": "in_progress",
                "modifiers": {},
                "station_bonuses": {},
            }

            # Apply crafting station bonuses
            available_stations = await self._get_available_crafting_stations(crafter_id)
            station_bonuses = await self._calculate_station_bonuses(
                recipe, available_stations
            )
            session["station_bonuses"] = station_bonuses

            self._active_crafting_sessions[session_id] = session

            logger.info(f"Started crafting session {session_id} for recipe {recipe_id}")

            return True, {
                "session_id": session_id,
                "recipe_name": recipe.name,
                "estimated_time": recipe.crafting_time,
                "total_steps": session["total_steps"],
                "station_bonuses": station_bonuses,
            }

        except Exception as e:
            logger.error(f"Error starting crafting session: {e}")
            return False, {"error": str(e)}

    async def process_crafting_step(
        self, session_id: str, step_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process a single step in the crafting process.

        Args:
            session_id: Active crafting session
            step_data: Data for the current crafting step

        Returns:
            Dict with step processing results
        """
        try:
            if session_id not in self._active_crafting_sessions:
                return {"error": f"Crafting session {session_id} not found"}

            session = self._active_crafting_sessions[session_id]

            if session["status"] != "in_progress":
                return {"error": f"Session {session_id} is not active"}

            recipe = session["recipe"]
            current_step = session["current_step"]
            total_steps = session["total_steps"]

            # Calculate step success probability
            step_success_probability = await self._calculate_step_success_probability(
                session, step_data
            )

            # Apply skill and station modifiers
            for modifier_name, modifier_func in self._crafting_modifiers.items():
                step_success_probability = await modifier_func(
                    step_success_probability, session, step_data
                )

            # Determine step success
            import random

            step_successful = random.random() < step_success_probability

            # Update progress
            step_progress = 1.0 / total_steps
            if step_successful:
                session["progress"] += step_progress
                session["current_step"] += 1
            else:
                # Failure can set back progress
                session["progress"] = max(
                    0.0, session["progress"] - step_progress * 0.5
                )

            # Check if crafting is complete
            crafting_complete = session["current_step"] >= total_steps
            crafting_failed = session["progress"] <= 0.0 and current_step > 2

            result = {
                "step_number": current_step + 1,
                "step_successful": step_successful,
                "progress": session["progress"],
                "crafting_complete": crafting_complete,
                "crafting_failed": crafting_failed,
            }

            if step_successful:
                result["message"] = f"Step {current_step + 1} completed successfully"

                # Apply step-specific effects
                step_effects = await self._apply_step_effects(session, step_data)
                result["step_effects"] = step_effects
            else:
                result["message"] = f"Step {current_step + 1} failed"
                result["failure_reason"] = await self._determine_failure_reason(
                    session, step_data
                )

            # Update session status
            if crafting_complete:
                session["status"] = "completed"
                result["final_result"] = await self._finalize_crafting(session)
            elif crafting_failed:
                session["status"] = "failed"
                result["failure_result"] = await self._handle_crafting_failure(session)

            return result

        except Exception as e:
            logger.error(f"Error processing crafting step: {e}")
            return {"error": str(e)}

    async def complete_crafting_session(
        self, session_id: str
    ) -> tuple[bool, dict[str, Any]]:
        """
        Complete crafting session and generate products.

        Args:
            session_id: Session to complete

        Returns:
            Tuple of (success, completion_result)
        """
        try:
            if session_id not in self._active_crafting_sessions:
                return False, {"error": f"Session {session_id} not found"}

            session = self._active_crafting_sessions[session_id]
            recipe = session["recipe"]

            if session["status"] != "completed":
                return False, {"error": "Session is not ready for completion"}

            # Calculate final success probability
            final_success_probability = (
                await self.calculate_crafting_success_probability(
                    recipe.recipe_id,
                    await self._get_crafter_skills(session["crafter_id"]),
                    {},  # Component quality would be calculated here
                )
            )

            # Apply session modifiers
            for bonus_name, bonus_value in session["station_bonuses"].items():
                if "success" in bonus_name:
                    final_success_probability += bonus_value

            # Determine final success
            import random

            crafting_successful = random.random() < final_success_probability

            completion_result = {
                "session_id": session_id,
                "recipe_name": recipe.name,
                "success": crafting_successful,
                "completion_time": asyncio.get_event_loop().time(),
                "total_time": asyncio.get_event_loop().time() - session["start_time"],
            }

            if crafting_successful:
                # Generate products
                products_created = await self._create_crafting_products(session)
                completion_result["products"] = products_created

                # Generate byproducts
                if recipe.byproducts:
                    byproducts_created = await self._create_crafting_byproducts(session)
                    completion_result["byproducts"] = byproducts_created

                # Award skill experience
                experience_gained = await self._award_crafting_experience(session)
                completion_result["experience"] = experience_gained

                completion_result["message"] = f"Successfully crafted {recipe.name}!"
            else:
                # Partial failure - might get some byproducts
                completion_result["message"] = f"Failed to craft {recipe.name}"

                # Chance for salvage materials
                salvage = await self._generate_failure_salvage(session)
                if salvage:
                    completion_result["salvage"] = salvage

            # Record in crafting history
            await self._record_crafting_attempt(session, completion_result)

            # Clean up session
            del self._active_crafting_sessions[session_id]

            return crafting_successful, completion_result

        except Exception as e:
            logger.error(f"Error completing crafting session: {e}")
            return False, {"error": str(e)}

    async def cancel_crafting_session(
        self, session_id: str, recovery_percentage: float = 0.5
    ) -> dict[str, Any]:
        """
        Cancel crafting session with partial component recovery.

        Args:
            session_id: Session to cancel
            recovery_percentage: Percentage of components to recover

        Returns:
            Dict with cancellation results
        """
        try:
            if session_id not in self._active_crafting_sessions:
                return {"error": f"Session {session_id} not found"}

            session = self._active_crafting_sessions[session_id]

            # Calculate recovery based on progress and recovery percentage
            progress = session["progress"]
            actual_recovery = recovery_percentage * (
                1.0 - progress * 0.5
            )  # Less recovery if more progress

            # Return components to inventories
            recovered_components = {}
            for component, reservation in session["reserved_components"].items():
                quantity_to_recover = int(reservation["quantity"] * actual_recovery)

                if quantity_to_recover > 0 and self.inventory:
                    success, _ = await self.inventory.add_item(
                        reservation["source"], component, quantity_to_recover
                    )

                    if success:
                        recovered_components[component] = quantity_to_recover

            # Record cancellation
            cancellation_result = {
                "session_id": session_id,
                "cancelled_at": asyncio.get_event_loop().time(),
                "progress_lost": progress,
                "components_recovered": recovered_components,
                "recovery_percentage": actual_recovery,
            }

            await self._record_crafting_cancellation(session, cancellation_result)

            # Clean up session
            del self._active_crafting_sessions[session_id]

            return cancellation_result

        except Exception as e:
            logger.error(f"Error cancelling crafting session: {e}")
            return {"error": str(e)}

    async def discover_recipe(
        self, components: list[str], context: dict[str, Any]
    ) -> CraftingRecipe | None:
        """
        Attempt to discover new recipe from available components.

        Args:
            components: Available components
            context: Current game context

        Returns:
            Discovered recipe or None
        """
        try:
            # Check existing discovery patterns
            for pattern_name, patterns in self._discovery_patterns.items():
                for pattern in patterns:
                    if await self._matches_discovery_pattern(components, pattern):
                        # Generate recipe from pattern
                        discovered_recipe = await self._generate_recipe_from_pattern(
                            pattern, components, context
                        )

                        if discovered_recipe:
                            # Add to registry
                            self._recipe_registry[discovered_recipe.recipe_id] = (
                                discovered_recipe
                            )

                            logger.info(
                                f"Discovered new recipe: {discovered_recipe.name}"
                            )
                            return discovered_recipe

            # Attempt dynamic discovery based on component properties
            dynamic_recipe = await self._attempt_dynamic_discovery(components, context)
            if dynamic_recipe:
                self._recipe_registry[dynamic_recipe.recipe_id] = dynamic_recipe
                return dynamic_recipe

            return None

        except Exception as e:
            logger.error(f"Error discovering recipe: {e}")
            return None

    async def validate_crafting_requirements(
        self, recipe_id: str, crafter_id: str, available_components: dict[str, int]
    ) -> tuple[bool, list[str]]:
        """
        Validate all requirements for crafting attempt.

        Args:
            recipe_id: Recipe to validate
            crafter_id: Entity attempting to craft
            available_components: Available components and quantities

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            if recipe_id not in self._recipe_registry:
                return False, [f"Recipe {recipe_id} not found"]

            recipe = self._recipe_registry[recipe_id]
            errors = []

            # Check skill requirements
            crafter_skills = await self._get_crafter_skills(crafter_id)
            for skill, required_level in recipe.required_skills.items():
                crafter_level = crafter_skills.get(skill, 0)
                if crafter_level < required_level:
                    errors.append(
                        f"Requires {skill} level {required_level}, have {crafter_level}"
                    )

            # Check component requirements
            for component, required_quantity in recipe.required_components.items():
                available_quantity = available_components.get(component, 0)
                if available_quantity < required_quantity:
                    errors.append(
                        f"Need {required_quantity} {component}, have {available_quantity}"
                    )

            # Check tool requirements
            available_tools = await self._get_available_tools(crafter_id)
            for required_tool in recipe.required_tools:
                if required_tool not in available_tools:
                    errors.append(f"Missing required tool: {required_tool}")

            # Check crafting station requirements
            available_stations = await self._get_available_crafting_stations(crafter_id)
            if recipe.crafting_stations:
                station_available = any(
                    station in available_stations
                    for station in recipe.crafting_stations
                )
                if not station_available:
                    errors.append(
                        f"Requires one of: {', '.join(recipe.crafting_stations)}"
                    )

            # Check energy requirements
            crafter_state = await self._get_crafter_state(crafter_id)
            current_energy = crafter_state.get("energy", 100)
            if current_energy < recipe.energy_cost:
                errors.append(
                    f"Insufficient energy: need {recipe.energy_cost}, have {current_energy}"
                )

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating crafting requirements: {e}")
            return False, [str(e)]

    async def calculate_crafting_success_probability(
        self,
        recipe_id: str,
        crafter_skills: dict[str, int],
        component_quality: dict[str, float],
    ) -> float:
        """
        Calculate probability of successful crafting.

        Args:
            recipe_id: Recipe being attempted
            crafter_skills: Crafter's skill levels
            component_quality: Quality of components being used

        Returns:
            Probability of success (0.0 to 1.0)
        """
        try:
            if recipe_id not in self._recipe_registry:
                return 0.0

            recipe = self._recipe_registry[recipe_id]

            # Base success chance from recipe
            success_probability = recipe.base_success_chance

            # Apply skill modifiers
            for skill, required_level in recipe.required_skills.items():
                crafter_level = crafter_skills.get(skill, 0)

                if crafter_level >= required_level:
                    # Bonus for higher skill
                    skill_bonus = (crafter_level - required_level) * 0.05
                    success_probability += skill_bonus
                else:
                    # Penalty for lower skill
                    skill_penalty = (required_level - crafter_level) * 0.1
                    success_probability -= skill_penalty

            # Apply component quality modifiers
            if component_quality:
                avg_quality = sum(component_quality.values()) / len(component_quality)
                quality_modifier = (
                    avg_quality - 0.5
                ) * 0.2  # Quality around 0.5 is neutral
                success_probability += quality_modifier

            # Apply complexity modifier
            complexity_modifiers = {
                CraftingComplexity.TRIVIAL: 0.2,
                CraftingComplexity.SIMPLE: 0.1,
                CraftingComplexity.MODERATE: 0.0,
                CraftingComplexity.COMPLEX: -0.1,
                CraftingComplexity.MASTER: -0.2,
                CraftingComplexity.LEGENDARY: -0.3,
            }

            complexity_modifier = complexity_modifiers.get(recipe.complexity, 0.0)
            success_probability += complexity_modifier

            # Clamp to valid range
            return max(0.0, min(1.0, success_probability))

        except Exception as e:
            logger.error(f"Error calculating crafting success probability: {e}")
            return 0.5

    async def get_available_recipes(
        self, crafter_id: str, available_components: list[str]
    ) -> list[CraftingRecipe]:
        """
        Get all recipes that could be attempted with available resources.

        Args:
            crafter_id: Entity to check recipes for
            available_components: List of available component IDs

        Returns:
            List of craftable recipes
        """
        try:
            craftable_recipes = []
            crafter_skills = await self._get_crafter_skills(crafter_id)
            available_tools = await self._get_available_tools(crafter_id)
            available_stations = await self._get_available_crafting_stations(crafter_id)

            for recipe in self._recipe_registry.values():
                # Check if recipe can be attempted
                can_craft = True

                # Check skill requirements
                for skill, required_level in recipe.required_skills.items():
                    if crafter_skills.get(skill, 0) < required_level:
                        can_craft = False
                        break

                if not can_craft:
                    continue

                # Check component requirements
                for component in recipe.required_components:
                    if component not in available_components:
                        can_craft = False
                        break

                if not can_craft:
                    continue

                # Check tool requirements
                for tool in recipe.required_tools:
                    if tool not in available_tools:
                        can_craft = False
                        break

                if not can_craft:
                    continue

                # Check station requirements
                if recipe.crafting_stations:
                    station_available = any(
                        station in available_stations
                        for station in recipe.crafting_stations
                    )
                    if not station_available:
                        can_craft = False

                if can_craft:
                    craftable_recipes.append(recipe)

            # Sort by complexity and success probability
            craftable_recipes.sort(
                key=lambda r: (r.complexity.value, -r.base_success_chance)
            )

            return craftable_recipes

        except Exception as e:
            logger.error(f"Error getting available recipes: {e}")
            return []

    async def enhance_crafting_with_modifiers(
        self, session_id: str, modifiers: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply temporary modifiers to ongoing crafting session.

        Args:
            session_id: Session to modify
            modifiers: Modifiers to apply

        Returns:
            Dict with modifier application results
        """
        try:
            if session_id not in self._active_crafting_sessions:
                return {"error": f"Session {session_id} not found"}

            session = self._active_crafting_sessions[session_id]

            # Apply modifiers
            for modifier_name, modifier_value in modifiers.items():
                session["modifiers"][modifier_name] = modifier_value

            return {
                "session_id": session_id,
                "modifiers_applied": modifiers,
                "total_modifiers": session["modifiers"],
            }

        except Exception as e:
            logger.error(f"Error enhancing crafting with modifiers: {e}")
            return {"error": str(e)}

    async def analyze_component_compatibility(
        self, components: list[str]
    ) -> dict[str, Any]:
        """
        Analyze how well components work together for crafting.

        Args:
            components: Components to analyze

        Returns:
            Dict with compatibility analysis
        """
        try:
            compatibility_matrix = {}
            overall_compatibility = 0.0
            total_pairs = 0

            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components[i + 1 :], i + 1):
                    compatibility = await self._calculate_component_compatibility(
                        comp1, comp2
                    )
                    compatibility_matrix[f"{comp1}+{comp2}"] = compatibility
                    overall_compatibility += compatibility
                    total_pairs += 1

            avg_compatibility = (
                overall_compatibility / total_pairs if total_pairs > 0 else 1.0
            )

            return {
                "components": components,
                "compatibility_matrix": compatibility_matrix,
                "overall_compatibility": avg_compatibility,
                "compatibility_rating": (
                    "excellent"
                    if avg_compatibility > 0.8
                    else (
                        "good"
                        if avg_compatibility > 0.6
                        else "fair" if avg_compatibility > 0.4 else "poor"
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing component compatibility: {e}")
            return {"error": str(e)}

    def register_recipe(self, recipe: CraftingRecipe) -> None:
        """
        Register a new crafting recipe.

        Args:
            recipe: Recipe to register
        """
        self._recipe_registry[recipe.recipe_id] = recipe

    def register_crafting_station(
        self, station_id: str, capabilities: dict[str, Any]
    ) -> None:
        """
        Register a new crafting station with its capabilities.

        Args:
            station_id: Unique identifier for the station
            capabilities: Station capabilities and bonuses
        """
        self._crafting_stations[station_id] = capabilities

    # Private helper methods

    async def _validate_basic_requirements(
        self, recipe: CraftingRecipe, crafter_id: str
    ) -> tuple[bool, list[str]]:
        """
        Validate basic crafting requirements (skills, tools, stations) without components.

        Args:
            recipe: Recipe to validate
            crafter_id: Entity attempting to craft

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            errors = []

            # Check skill requirements
            crafter_skills = await self._get_crafter_skills(crafter_id)
            for skill, required_level in recipe.required_skills.items():
                crafter_level = crafter_skills.get(skill, 0)
                if crafter_level < required_level:
                    errors.append(
                        f"Requires {skill} level {required_level}, have {crafter_level}"
                    )

            # Check tool requirements
            available_tools = await self._get_available_tools(crafter_id)
            for required_tool in recipe.required_tools:
                if required_tool not in available_tools:
                    errors.append(f"Missing required tool: {required_tool}")

            # Check crafting station requirements
            available_stations = await self._get_available_crafting_stations(crafter_id)
            if recipe.crafting_stations:
                station_available = any(
                    station in available_stations
                    for station in recipe.crafting_stations
                )
                if not station_available:
                    errors.append(
                        f"Requires one of: {', '.join(recipe.crafting_stations)}"
                    )

            # Check energy requirements
            crafter_state = await self._get_crafter_state(crafter_id)
            current_energy = crafter_state.get("energy", 100)
            if current_energy < recipe.energy_cost:
                errors.append(
                    f"Insufficient energy: need {recipe.energy_cost}, have {current_energy}"
                )

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating basic crafting requirements: {e}")
            return False, [str(e)]

    async def _rollback_component_reservations(
        self, reserved_components: dict[str, dict[str, Any]]
    ) -> None:
        """Rollback component reservations in case of failure."""
        for component, reservation in reserved_components.items():
            if self.inventory:
                await self.inventory.add_item(
                    reservation["source"], component, reservation["quantity"]
                )

    def _calculate_crafting_steps(self, recipe: CraftingRecipe) -> int:
        """Calculate number of steps required for recipe."""
        complexity_steps = {
            CraftingComplexity.TRIVIAL: 1,
            CraftingComplexity.SIMPLE: 2,
            CraftingComplexity.MODERATE: 3,
            CraftingComplexity.COMPLEX: 5,
            CraftingComplexity.MASTER: 8,
            CraftingComplexity.LEGENDARY: 12,
        }
        return complexity_steps.get(recipe.complexity, 3)

    async def _get_available_crafting_stations(self, crafter_id: str) -> list[str]:
        """Get list of available crafting stations for crafter."""
        # This would check the crafter's location and available stations
        return ["forge", "basic_forge", "workbench", "alchemy_table"]

    async def _calculate_station_bonuses(
        self, recipe: CraftingRecipe, available_stations: list[str]
    ) -> dict[str, float]:
        """Calculate bonuses from available crafting stations."""
        bonuses = {}

        for station in available_stations:
            if station in recipe.crafting_stations:
                station_capabilities = self._crafting_stations.get(station, {})

                for bonus_type, bonus_value in station_capabilities.items():
                    if bonus_type not in bonuses:
                        bonuses[bonus_type] = 0.0
                    bonuses[bonus_type] += bonus_value

        return bonuses

    async def _calculate_step_success_probability(
        self, session: dict[str, Any], step_data: dict[str, Any]
    ) -> float:
        """Calculate success probability for a crafting step."""
        recipe = session["recipe"]
        base_probability = recipe.base_success_chance

        # Apply step-specific modifiers
        step_difficulty = step_data.get("difficulty", 1.0)
        step_probability = base_probability / step_difficulty

        return max(0.1, min(0.95, step_probability))

    async def _apply_step_effects(
        self, session: dict[str, Any], step_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply effects of successful crafting step."""
        return {"step_completed": True}

    async def _determine_failure_reason(
        self, session: dict[str, Any], step_data: dict[str, Any]
    ) -> str:
        """Determine reason for step failure."""
        failure_reasons = [
            "Insufficient skill",
            "Component contamination",
            "Tool malfunction",
            "Environmental interference",
            "Timing error",
        ]

        import random

        return random.choice(failure_reasons)

    async def _finalize_crafting(self, session: dict[str, Any]) -> dict[str, Any]:
        """Finalize completed crafting session."""
        return {"status": "completed", "ready_for_completion": True}

    async def _handle_crafting_failure(self, session: dict[str, Any]) -> dict[str, Any]:
        """Handle failed crafting session."""
        return {"status": "failed", "components_lost": True}

    async def _get_crafter_skills(self, crafter_id: str) -> dict[str, int]:
        """Get skill levels for crafter."""
        if self.skills:
            return await self.skills.get_skills(crafter_id)

        # Default skills for testing
        return {"crafting": 5, "smithing": 3, "woodworking": 4}

    async def _get_available_tools(self, crafter_id: str) -> list[str]:
        """Get available tools for crafter."""
        # This would check the crafter's inventory and nearby tools
        return ["hammer", "anvil", "chisel", "saw"]

    async def _get_crafter_state(self, crafter_id: str) -> dict[str, Any]:
        """Get current state of crafter."""
        return {"energy": 100, "focus": 80}

    async def _create_crafting_products(
        self, session: dict[str, Any]
    ) -> dict[str, int]:
        """Create products from successful crafting."""
        recipe = session["recipe"]
        products_created = {}

        for product, quantity in recipe.products.items():
            # Quality could be affected by skill and modifiers
            products_created[product] = quantity

        return products_created

    async def _create_crafting_byproducts(
        self, session: dict[str, Any]
    ) -> dict[str, int]:
        """Create byproducts from crafting."""
        recipe = session["recipe"]
        return recipe.byproducts

    async def _award_crafting_experience(
        self, session: dict[str, Any]
    ) -> dict[str, float]:
        """Award skill experience for successful crafting."""
        recipe = session["recipe"]
        return recipe.skill_experience

    async def _generate_failure_salvage(
        self, session: dict[str, Any]
    ) -> dict[str, int] | None:
        """Generate salvage materials from failed crafting."""
        # Chance to get some materials back
        import random

        if random.random() < 0.3:  # 30% chance for salvage
            return {"scrap_material": 1}
        return None

    async def _record_crafting_attempt(
        self, session: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Record crafting attempt in history."""
        crafter_id = session["crafter_id"]

        if crafter_id not in self._crafting_history:
            self._crafting_history[crafter_id] = []

        record = {
            "timestamp": asyncio.get_event_loop().time(),
            "recipe_id": session["recipe"].recipe_id,
            "success": result["success"],
            "result": result,
        }

        self._crafting_history[crafter_id].append(record)

        # Limit history size
        max_history = 100
        if len(self._crafting_history[crafter_id]) > max_history:
            self._crafting_history[crafter_id] = self._crafting_history[crafter_id][
                -max_history:
            ]

    async def _record_crafting_cancellation(
        self, session: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Record crafting cancellation."""
        await self._record_crafting_attempt(
            session,
            {
                "success": False,
                "cancelled": True,
                "cancellation_data": result,
            },
        )

    async def _matches_discovery_pattern(
        self, components: list[str], pattern: dict[str, Any]
    ) -> bool:
        """Check if components match a discovery pattern."""
        required_components = pattern.get("components", [])
        return all(comp in components for comp in required_components)

    async def _generate_recipe_from_pattern(
        self, pattern: dict[str, Any], components: list[str], context: dict[str, Any]
    ) -> CraftingRecipe | None:
        """Generate a recipe from a discovery pattern."""
        recipe_id = f"discovered_{uuid.uuid4().hex[:8]}"

        return CraftingRecipe(
            recipe_id=recipe_id,
            name=pattern.get("name", "Discovered Recipe"),
            description=pattern.get("description", "A newly discovered recipe"),
            required_components=dict.fromkeys(components, 1),
            optional_components={},
            required_tools=pattern.get("tools", []),
            required_skills=pattern.get("skills", {}),
            crafting_stations=pattern.get("stations", []),
            complexity=CraftingComplexity.SIMPLE,
            base_success_chance=0.7,
            crafting_time=300.0,
            energy_cost=20.0,
            products=pattern.get("products", {"unknown_item": 1}),
            byproducts={},
            skill_experience={"crafting": 5.0},
        )

    async def _attempt_dynamic_discovery(
        self, components: list[str], context: dict[str, Any]
    ) -> CraftingRecipe | None:
        """Attempt dynamic recipe discovery based on component analysis."""
        # This would use AI/ML or heuristics to suggest new recipes
        return None

    async def _calculate_component_compatibility(self, comp1: str, comp2: str) -> float:
        """Calculate compatibility between two components."""
        # This would analyze component properties and relationships
        return 0.7  # Default moderate compatibility

    def _initialize_recipes(self) -> None:
        """Initialize default crafting recipes."""

        # Basic sword recipe
        basic_sword = CraftingRecipe(
            recipe_id="basic_sword",
            name="Basic Iron Sword",
            description="A simple iron sword for beginners",
            required_components={"iron_ingot": 2, "wood_handle": 1},
            optional_components={"leather_wrap": 1},
            required_tools=["hammer", "anvil"],
            required_skills={"smithing": 3},
            crafting_stations=["forge"],
            complexity=CraftingComplexity.SIMPLE,
            base_success_chance=0.8,
            crafting_time=1800.0,  # 30 minutes
            energy_cost=25.0,
            products={"iron_sword": 1},
            byproducts={"metal_shavings": 1},
            skill_experience={"smithing": 10.0},
        )

        # Healing potion recipe
        healing_potion = CraftingRecipe(
            recipe_id="healing_potion",
            name="Basic Healing Potion",
            description="A potion that restores health",
            required_components={"herbs": 3, "water": 1, "glass_vial": 1},
            optional_components={"honey": 1},
            required_tools=["mortar_pestle"],
            required_skills={"alchemy": 2},
            crafting_stations=["alchemy_table"],
            complexity=CraftingComplexity.SIMPLE,
            base_success_chance=0.75,
            crafting_time=600.0,  # 10 minutes
            energy_cost=15.0,
            products={"healing_potion": 1},
            byproducts={},
            skill_experience={"alchemy": 5.0},
        )

        self._recipe_registry[basic_sword.recipe_id] = basic_sword
        self._recipe_registry[healing_potion.recipe_id] = healing_potion

        # Initialize discovery patterns
        self._discovery_patterns["metal_working"] = [
            {
                "components": ["iron_ingot", "wood_handle"],
                "name": "Discovered Tool",
                "products": {"iron_tool": 1},
                "tools": ["hammer"],
                "skills": {"smithing": 2},
                "stations": ["forge"],
            }
        ]

        # Initialize crafting stations
        self._crafting_stations["forge"] = {
            "success_bonus": 0.1,
            "quality_bonus": 0.15,
            "time_reduction": 0.2,
        }

        self._crafting_stations["alchemy_table"] = {
            "success_bonus": 0.05,
            "purity_bonus": 0.2,
            "ingredient_efficiency": 0.1,
        }
