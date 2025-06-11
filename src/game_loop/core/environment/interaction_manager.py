"""
Environment Interaction Manager for handling interactions with environmental elements.

This module provides object manipulation, container interactions, tool usage,
and environmental puzzle mechanics for the game world.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionResult,
    PhysicalActionType,
)

logger = logging.getLogger(__name__)


class InteractionType:
    """Constants for interaction types."""

    EXAMINE = "examine"
    MANIPULATE = "manipulate"
    CONTAINER = "container"
    TOOL_USE = "tool_use"
    PUZZLE = "puzzle"
    MECHANISM = "mechanism"


class EnvironmentInteractionManager:
    """
    Manage interactions between entities and environmental elements.

    This class handles object manipulation, container interactions, tool usage,
    environmental state tracking, and puzzle mechanics.
    """

    def __init__(
        self,
        world_state_manager: Any = None,
        object_manager: Any = None,
        physics_engine: Any = None,
    ):
        """
        Initialize the environment interaction manager.

        Args:
            world_state_manager: Manager for world state access and updates
            object_manager: Manager for object data and state
            physics_engine: Physics engine for interaction constraints
        """
        self.world_state = world_state_manager
        self.objects = object_manager
        self.physics = physics_engine
        self._interaction_handlers: dict[str, Callable] = {}
        self._environmental_states: dict[str, dict[str, Any]] = {}
        self._interaction_history: list[dict[str, Any]] = []
        self._initialize_handlers()

    async def process_environment_interaction(
        self,
        player_id: str,
        target_entity: str,
        interaction_type: str,
        context: dict[str, Any],
    ) -> PhysicalActionResult:
        """
        Process interaction with environmental element.

        Args:
            player_id: ID of the player performing the interaction
            target_entity: Environmental entity to interact with
            interaction_type: Type of interaction to perform
            context: Current game context

        Returns:
            PhysicalActionResult containing interaction outcome
        """
        try:
            # Validate interaction requirements
            is_valid, requirements_error = await self.validate_interaction_requirements(
                interaction_type, target_entity, context.get("player_state", {})
            )

            if not is_valid:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[target_entity],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Interaction failed.",
                    error_message=(
                        requirements_error[0] if requirements_error else "Unknown error"
                    ),
                )

            # Get interaction handler
            handler = self._interaction_handlers.get(interaction_type)
            if not handler:
                return await self._handle_generic_interaction(
                    player_id, target_entity, interaction_type, context
                )

            # Execute specific interaction
            result = await handler(player_id, target_entity, context)

            # Update interaction history
            await self._record_interaction(
                player_id, target_entity, interaction_type, result
            )

            # Update environmental state
            if result.success:
                await self.update_environmental_state(
                    target_entity, result.state_changes
                )

            return result

        except Exception as e:
            logger.error(f"Error processing environment interaction: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[target_entity],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Interaction failed due to an error.",
                error_message=str(e),
            )

    async def validate_interaction_requirements(
        self,
        interaction_type: str,
        target_entity: str,
        player_state: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate that player meets requirements for interaction.

        Args:
            interaction_type: Type of interaction
            target_entity: Target entity for interaction
            player_state: Current player state

        Returns:
            Tuple of (is_valid, list_of_requirement_errors)
        """
        try:
            errors = []

            # Check if player_state is provided and valid
            if not player_state or (
                isinstance(player_state, dict) and len(player_state) == 0
            ):
                errors.append("Player state is required for interactions.")
                return False, errors

            # Check energy requirements
            current_energy = player_state.get("energy", 100)
            min_energy_required = 5  # Minimum energy for any interaction

            if interaction_type in ["push", "pull", "lift", "move"]:
                min_energy_required = 10  # Physical actions require more energy

            if current_energy < min_energy_required:
                errors.append(
                    f"Insufficient energy for {interaction_type}. Need {min_energy_required}, have {current_energy}."
                )

            # Check basic accessibility
            if not await self._validate_object_accessibility(
                target_entity, player_state.get("position", {})
            ):
                errors.append(f"{target_entity} is not within reach.")

            # Check interaction-specific requirements
            if interaction_type == InteractionType.TOOL_USE:
                required_tool = await self._get_required_tool(
                    target_entity, interaction_type
                )
                if required_tool and required_tool not in player_state.get(
                    "inventory", []
                ):
                    errors.append(
                        f"You need a {required_tool} to interact with {target_entity}."
                    )

            # Check skill requirements
            required_skills = await self._get_interaction_skill_requirements(
                interaction_type, target_entity
            )
            for skill, required_level in required_skills.items():
                player_level = player_state.get("skills", {}).get(skill, 0)
                if player_level < required_level:
                    errors.append(
                        f"You need {skill} level {required_level} to interact with {target_entity}."
                    )

            # Check state requirements
            entity_state = await self._get_entity_state(target_entity)
            if entity_state.get("broken", False):
                errors.append(f"{target_entity} is broken and cannot be used.")

            if (
                entity_state.get("locked", False)
                and interaction_type != InteractionType.EXAMINE
            ):
                errors.append(f"{target_entity} is locked.")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating interaction requirements: {e}")
            return False, [f"Validation error: {str(e)}"]

    async def execute_object_manipulation(
        self, object_id: str, manipulation_type: str, player_state: dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Execute manipulation of environmental objects.

        Args:
            object_id: ID of the object to manipulate
            manipulation_type: Type of manipulation (push, pull, lift, etc.)
            player_state: Current player state

        Returns:
            PhysicalActionResult with manipulation outcome
        """
        try:
            # Get object properties
            object_info = await self._get_object_info(object_id)
            if not object_info:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[object_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Object manipulation failed.",
                    error_message=f"Object {object_id} not found.",
                )

            # Calculate manipulation difficulty
            difficulty = await self._calculate_interaction_difficulty(
                manipulation_type, object_id
            )

            # Calculate energy and time costs
            mass = object_info.get("mass", 10.0)
            energy_cost = self._calculate_manipulation_energy_cost(
                manipulation_type, mass, difficulty
            )
            time_cost = self._calculate_manipulation_time_cost(
                manipulation_type, mass, difficulty
            )

            # Check if manipulation is possible
            max_strength = player_state.get("strength", 50)
            required_strength = mass * difficulty

            if required_strength > max_strength:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[object_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description=f"The {object_id} is too heavy to {manipulation_type}.",
                    error_message=f"Insufficient strength: {required_strength} required, {max_strength} available",
                )

            # Execute manipulation
            state_changes = await self._apply_manipulation_effects(
                object_id, manipulation_type, player_state
            )

            # Generate description
            description = await self._generate_manipulation_description(
                object_id, manipulation_type, state_changes
            )

            return PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[object_id],
                state_changes=state_changes,
                energy_cost=energy_cost,
                time_elapsed=time_cost,
                side_effects=await self._get_manipulation_side_effects(
                    manipulation_type, object_id
                ),
                description=description,
            )

        except Exception as e:
            logger.error(f"Error executing object manipulation: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[object_id],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Object manipulation failed due to an error.",
                error_message=str(e),
            )

    async def handle_container_interactions(
        self,
        container_id: str,
        action: str,
        item_id: str | None,
        context: dict[str, Any],
    ) -> PhysicalActionResult:
        """
        Handle interactions with containers (chests, doors, etc.).

        Args:
            container_id: ID of the container
            action: Action to perform (open, close, put, take)
            item_id: Optional item ID for put/take actions
            context: Current game context

        Returns:
            PhysicalActionResult with container interaction outcome
        """
        try:
            container_state = await self._get_entity_state(container_id)
            player_state = context.get("player_state", {})

            # Handle different container actions
            if action == "open":
                return await self._handle_container_open(
                    container_id, container_state, player_state
                )
            elif action == "close":
                return await self._handle_container_close(
                    container_id, container_state, player_state
                )
            elif action == "put" and item_id:
                return await self._handle_container_put(
                    container_id, item_id, container_state, player_state
                )
            elif action == "take" and item_id:
                return await self._handle_container_take(
                    container_id, item_id, container_state, player_state
                )
            else:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[container_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Invalid container action.",
                    error_message=f"Unknown action: {action}",
                )

        except Exception as e:
            logger.error(f"Error handling container interaction: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[container_id],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Container interaction failed due to an error.",
                error_message=str(e),
            )

    async def process_tool_usage(
        self, tool_id: str, target_id: str, action: str, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Process usage of tools on environmental targets.

        Args:
            tool_id: ID of the tool being used
            target_id: ID of the target entity
            action: Action being performed with the tool
            context: Current game context

        Returns:
            PhysicalActionResult with tool usage outcome
        """
        try:
            player_state = context.get("player_state", {})

            # Validate tool ownership
            if tool_id not in player_state.get("inventory", []):
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[tool_id, target_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Tool usage failed.",
                    error_message=f"You don't have {tool_id} in your inventory.",
                )

            # Get tool and target information
            tool_info = await self._get_tool_info(tool_id)
            target_info = await self._get_object_info(target_id)

            if not tool_info or not target_info:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[tool_id, target_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Tool usage failed.",
                    error_message="Tool or target not found.",
                )

            # Check tool compatibility
            compatible_actions = tool_info.get("compatible_actions", [])
            if action not in compatible_actions:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[tool_id, target_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description=f"You cannot {action} {target_id} with {tool_id}.",
                    error_message=f"Tool {tool_id} is not compatible with action {action}",
                )

            # Calculate tool usage effects
            effectiveness = tool_info.get("effectiveness", {}).get(action, 1.0)
            energy_cost = (
                15.0 / effectiveness
            )  # More effective tools require less energy
            time_cost = 10.0 / effectiveness

            # Apply tool usage effects
            state_changes = await self._apply_tool_usage_effects(
                tool_id, target_id, action, effectiveness
            )

            # Check for tool wear
            tool_wear = await self._calculate_tool_wear(tool_id, action, effectiveness)
            if tool_wear > 0:
                state_changes[f"{tool_id}_durability"] = -tool_wear

            description = f"You use {tool_id} to {action} {target_id}."
            if effectiveness > 1.5:
                description += " The tool makes the task much easier."
            elif effectiveness < 0.7:
                description += " The tool is not very effective for this task."

            return PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[tool_id, target_id],
                state_changes=state_changes,
                energy_cost=energy_cost,
                time_elapsed=time_cost,
                side_effects=await self._get_tool_usage_side_effects(
                    tool_id, target_id, action
                ),
                description=description,
            )

        except Exception as e:
            logger.error(f"Error processing tool usage: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[tool_id, target_id],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Tool usage failed due to an error.",
                error_message=str(e),
            )

    async def handle_environmental_puzzles(
        self, puzzle_element: str, action: str, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Handle interactions with puzzle elements.

        Args:
            puzzle_element: ID of the puzzle element
            action: Action being performed on the puzzle
            context: Current game context

        Returns:
            PhysicalActionResult with puzzle interaction outcome
        """
        try:
            puzzle_state = await self._get_puzzle_state(puzzle_element)
            player_state = context.get("player_state", {})

            # Check if puzzle is already solved
            if puzzle_state.get("solved", False):
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MANIPULATION,
                    affected_entities=[puzzle_element],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="This puzzle has already been solved.",
                    error_message="Puzzle already completed.",
                )

            # Process puzzle action
            puzzle_result = await self._process_puzzle_action(
                puzzle_element, action, puzzle_state, player_state
            )

            # Check if action solves the puzzle
            is_solved = await self._check_puzzle_completion(
                puzzle_element, puzzle_result["new_state"]
            )

            state_changes = puzzle_result["state_changes"]
            if is_solved:
                state_changes[f"{puzzle_element}_solved"] = True
                # Apply puzzle completion rewards
                rewards = await self._get_puzzle_rewards(puzzle_element)
                state_changes.update(rewards)

            description = puzzle_result["description"]
            if is_solved:
                description += " You hear a satisfying click as the puzzle is solved!"

            return PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[puzzle_element],
                state_changes=state_changes,
                energy_cost=5.0,
                time_elapsed=8.0,
                side_effects=puzzle_result.get("side_effects", []),
                description=description,
            )

        except Exception as e:
            logger.error(f"Error handling environmental puzzle: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[puzzle_element],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Puzzle interaction failed due to an error.",
                error_message=str(e),
            )

    async def update_environmental_state(
        self, entity_id: str, state_changes: dict[str, Any]
    ) -> None:
        """
        Update environmental entity state.

        Args:
            entity_id: ID of the environmental entity
            state_changes: Changes to apply to the entity state
        """
        try:
            if entity_id not in self._environmental_states:
                self._environmental_states[entity_id] = {}

            # Apply state changes
            for key, value in state_changes.items():
                if key.startswith(entity_id):
                    # Extract the actual state key
                    state_key = key.replace(f"{entity_id}_", "")
                    self._environmental_states[entity_id][state_key] = value

            # Update in world state if available
            if self.world_state:
                # In a full implementation, this would update the world state
                logger.info(
                    f"Environmental state updated for {entity_id}: {state_changes}"
                )

        except Exception as e:
            logger.error(f"Error updating environmental state: {e}")

    async def check_interaction_side_effects(
        self,
        interaction_type: str,
        entities: list[str],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Check for side effects of environmental interactions.

        Args:
            interaction_type: Type of interaction performed
            entities: Entities involved in the interaction
            context: Current game context

        Returns:
            List of side effect descriptions
        """
        try:
            side_effects = []

            for entity in entities:
                entity_effects = await self._get_entity_interaction_effects(
                    entity, interaction_type
                )
                side_effects.extend(entity_effects)

            return side_effects

        except Exception as e:
            logger.error(f"Error checking interaction side effects: {e}")
            return []

    async def get_interaction_options(
        self, entity_id: str, player_state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Get available interaction options for an entity.

        Args:
            entity_id: ID of the entity
            player_state: Current player state

        Returns:
            List of available interaction options
        """
        try:
            options = []
            entity_info = await self._get_object_info(entity_id)

            if not entity_info:
                return options

            # Basic interactions available for all objects
            options.append(
                {
                    "action": "examine",
                    "description": f"Examine the {entity_id}",
                    "requirements": [],
                }
            )

            # Add interaction options based on entity type
            entity_type = entity_info.get("type", "object")

            if entity_type == "container":
                entity_state = await self._get_entity_state(entity_id)
                if entity_state.get("open", False):
                    options.append(
                        {
                            "action": "close",
                            "description": f"Close the {entity_id}",
                            "requirements": [],
                        }
                    )
                else:
                    options.append(
                        {
                            "action": "open",
                            "description": f"Open the {entity_id}",
                            "requirements": entity_state.get("open_requirements", []),
                        }
                    )

            if entity_info.get("moveable", False):
                options.extend(
                    [
                        {
                            "action": "push",
                            "description": f"Push the {entity_id}",
                            "requirements": [
                                f"strength >= {entity_info.get('mass', 10)}"
                            ],
                        },
                        {
                            "action": "pull",
                            "description": f"Pull the {entity_id}",
                            "requirements": [
                                f"strength >= {entity_info.get('mass', 10)}"
                            ],
                        },
                    ]
                )

            return options

        except Exception as e:
            logger.error(f"Error getting interaction options: {e}")
            return []

    def register_interaction_handler(
        self, interaction_type: str, handler: Callable
    ) -> None:
        """
        Register handler for specific interaction type.

        Args:
            interaction_type: Type of interaction to handle
            handler: Handler function for the interaction
        """
        self._interaction_handlers[interaction_type] = handler

    # Private helper methods

    async def _validate_object_accessibility(
        self, object_id: str, player_position: dict[str, Any]
    ) -> bool:
        """Check if object is within reach for interaction."""
        # In a full implementation, this would check actual distances
        return True

    async def _calculate_interaction_difficulty(
        self, interaction_type: str, target_entity: str
    ) -> float:
        """Calculate difficulty level for interaction."""
        base_difficulties = {
            "push": 1.2,
            "pull": 1.1,
            "lift": 1.5,
            "open": 0.8,
            "close": 0.7,
            "examine": 0.1,
        }
        return base_difficulties.get(interaction_type, 1.0)

    async def _apply_wear_and_tear(
        self, entity_id: str, usage_intensity: float
    ) -> None:
        """Apply wear and tear effects to frequently used objects."""
        if entity_id not in self._environmental_states:
            self._environmental_states[entity_id] = {"durability": 100.0}

        current_durability = self._environmental_states[entity_id].get(
            "durability", 100.0
        )
        wear_amount = usage_intensity * 2.0  # 2% wear per usage point
        new_durability = max(0.0, current_durability - wear_amount)

        self._environmental_states[entity_id]["durability"] = new_durability

        if new_durability <= 0:
            self._environmental_states[entity_id]["broken"] = True

    def _initialize_handlers(self) -> None:
        """Initialize default interaction handlers."""
        self._interaction_handlers[InteractionType.EXAMINE] = (
            self._handle_examine_interaction
        )
        self._interaction_handlers[InteractionType.CONTAINER] = (
            self._handle_container_interaction
        )
        self._interaction_handlers[InteractionType.TOOL_USE] = (
            self._handle_tool_use_interaction
        )

    # Additional helper methods (simplified implementations)

    async def _get_object_info(self, object_id: str) -> dict[str, Any] | None:
        """Get object information."""
        # Simulate object data
        return {
            "id": object_id,
            "type": "object",
            "mass": 10.0,
            "moveable": True,
            "description": f"A {object_id}",
        }

    async def _get_entity_state(self, entity_id: str) -> dict[str, Any]:
        """Get current state of an entity."""
        return self._environmental_states.get(entity_id, {})

    async def _handle_examine_interaction(
        self, player_id: str, target_entity: str, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """Handle examination interactions."""
        description = f"You examine the {target_entity} carefully."
        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=[target_entity],
            state_changes={},
            energy_cost=1.0,
            time_elapsed=2.0,
            side_effects=[],
            description=description,
        )

    async def _handle_container_interaction(
        self, player_id: str, target_entity: str, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """Handle container interactions."""
        return await self.handle_container_interactions(
            target_entity, "examine", None, context
        )

    async def _handle_tool_use_interaction(
        self, player_id: str, target_entity: str, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """Handle tool use interactions."""
        # This would need more context about which tool is being used
        description = f"You attempt to use a tool on {target_entity}."
        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=[target_entity],
            state_changes={},
            energy_cost=5.0,
            time_elapsed=5.0,
            side_effects=[],
            description=description,
        )

    # Placeholder implementations for additional helper methods
    async def _get_required_tool(
        self, target_entity: str, interaction_type: str
    ) -> str | None:
        return None

    async def _get_interaction_skill_requirements(
        self, interaction_type: str, target_entity: str
    ) -> dict[str, int]:
        return {}

    async def _handle_generic_interaction(
        self,
        player_id: str,
        target_entity: str,
        interaction_type: str,
        context: dict[str, Any],
    ) -> PhysicalActionResult:
        description = f"You {interaction_type} the {target_entity}."
        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=[target_entity],
            state_changes={},
            energy_cost=5.0,
            time_elapsed=3.0,
            side_effects=[],
            description=description,
        )

    async def _record_interaction(
        self,
        player_id: str,
        target_entity: str,
        interaction_type: str,
        result: PhysicalActionResult,
    ) -> None:
        self._interaction_history.append(
            {
                "player_id": player_id,
                "target_entity": target_entity,
                "interaction_type": interaction_type,
                "success": result.success,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def _calculate_manipulation_energy_cost(
        self, manipulation_type: str, mass: float, difficulty: float
    ) -> float:
        base_costs = {"push": 8.0, "pull": 7.0, "lift": 12.0, "drag": 6.0}
        base_cost = base_costs.get(manipulation_type, 8.0)
        return base_cost * (mass / 10.0) * difficulty

    def _calculate_manipulation_time_cost(
        self, manipulation_type: str, mass: float, difficulty: float
    ) -> float:
        base_times = {"push": 5.0, "pull": 4.0, "lift": 8.0, "drag": 6.0}
        base_time = base_times.get(manipulation_type, 5.0)
        return base_time * (1.0 + mass / 20.0) * difficulty

    async def _apply_manipulation_effects(
        self, object_id: str, manipulation_type: str, player_state: dict[str, Any]
    ) -> dict[str, Any]:
        return {f"{object_id}_position": f"{manipulation_type}_applied"}

    async def _generate_manipulation_description(
        self, object_id: str, manipulation_type: str, state_changes: dict[str, Any]
    ) -> str:
        return f"You {manipulation_type} the {object_id}."

    async def _get_manipulation_side_effects(
        self, manipulation_type: str, object_id: str
    ) -> list[str]:
        return []

    async def _handle_container_open(
        self,
        container_id: str,
        container_state: dict[str, Any],
        player_state: dict[str, Any],
    ) -> PhysicalActionResult:
        if container_state.get("locked", False):
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.OPENING,
                affected_entities=[container_id],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description=f"The {container_id} is locked.",
                error_message="Container is locked.",
            )

        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.OPENING,
            affected_entities=[container_id],
            state_changes={f"{container_id}_open": True},
            energy_cost=3.0,
            time_elapsed=2.0,
            side_effects=[],
            description=f"You open the {container_id}.",
        )

    async def _handle_container_close(
        self,
        container_id: str,
        container_state: dict[str, Any],
        player_state: dict[str, Any],
    ) -> PhysicalActionResult:
        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.CLOSING,
            affected_entities=[container_id],
            state_changes={f"{container_id}_open": False},
            energy_cost=2.0,
            time_elapsed=1.0,
            side_effects=[],
            description=f"You close the {container_id}.",
        )

    async def _handle_container_put(
        self,
        container_id: str,
        item_id: str,
        container_state: dict[str, Any],
        player_state: dict[str, Any],
    ) -> PhysicalActionResult:
        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=[container_id, item_id],
            state_changes={f"{item_id}_location": container_id},
            energy_cost=2.0,
            time_elapsed=1.5,
            side_effects=[],
            description=f"You put {item_id} in the {container_id}.",
        )

    async def _handle_container_take(
        self,
        container_id: str,
        item_id: str,
        container_state: dict[str, Any],
        player_state: dict[str, Any],
    ) -> PhysicalActionResult:
        return PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MANIPULATION,
            affected_entities=[container_id, item_id],
            state_changes={f"{item_id}_location": "inventory"},
            energy_cost=2.0,
            time_elapsed=1.5,
            side_effects=[],
            description=f"You take {item_id} from the {container_id}.",
        )

    async def _get_tool_info(self, tool_id: str) -> dict[str, Any] | None:
        return {
            "id": tool_id,
            "compatible_actions": ["cut", "dig", "repair"],
            "effectiveness": {"cut": 1.5, "dig": 1.2},
        }

    async def _apply_tool_usage_effects(
        self, tool_id: str, target_id: str, action: str, effectiveness: float
    ) -> dict[str, Any]:
        return {f"{target_id}_{action}": effectiveness}

    async def _calculate_tool_wear(
        self, tool_id: str, action: str, effectiveness: float
    ) -> float:
        return 2.0 / effectiveness  # Better tools wear less

    async def _get_tool_usage_side_effects(
        self, tool_id: str, target_id: str, action: str
    ) -> list[str]:
        return []

    async def _get_puzzle_state(self, puzzle_element: str) -> dict[str, Any]:
        return self._environmental_states.get(
            puzzle_element, {"solved": False, "attempts": 0}
        )

    async def _process_puzzle_action(
        self,
        puzzle_element: str,
        action: str,
        puzzle_state: dict[str, Any],
        player_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "new_state": {"attempts": puzzle_state.get("attempts", 0) + 1},
            "state_changes": {
                f"{puzzle_element}_attempts": puzzle_state.get("attempts", 0) + 1
            },
            "description": f"You {action} the puzzle mechanism.",
            "side_effects": [],
        }

    async def _check_puzzle_completion(
        self, puzzle_element: str, new_state: dict[str, Any]
    ) -> bool:
        # Simple puzzle: solved after 3 attempts
        return new_state.get("attempts", 0) >= 3

    async def _get_puzzle_rewards(self, puzzle_element: str) -> dict[str, Any]:
        return {"experience": 50, "puzzle_solved": puzzle_element}

    async def _get_entity_interaction_effects(
        self, entity: str, interaction_type: str
    ) -> list[dict[str, Any]]:
        return []
