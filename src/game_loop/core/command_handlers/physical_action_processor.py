"""
Physical Action Processor for handling physical actions in the game world.

This module provides the core processor for handling all physical actions including
movement, environment interaction, and spatial manipulation.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.console import Console

from game_loop.core.actions.types import ActionClassification

logger = logging.getLogger(__name__)


class PhysicalActionType(Enum):
    """Types of physical actions that can be performed."""

    MOVEMENT = "movement"
    MANIPULATION = "manipulation"
    CLIMBING = "climbing"
    JUMPING = "jumping"
    PUSHING = "pushing"
    PULLING = "pulling"
    LIFTING = "lifting"
    OPENING = "opening"
    CLOSING = "closing"
    BREAKING = "breaking"
    BUILDING = "building"


@dataclass
class PhysicalActionResult:
    """Result of a physical action execution."""

    success: bool
    action_type: PhysicalActionType
    affected_entities: list[str]
    state_changes: dict[str, Any]
    energy_cost: float
    time_elapsed: float
    side_effects: list[str]
    description: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "success": self.success,
            "action_type": self.action_type.value,
            "affected_entities": self.affected_entities,
            "state_changes": self.state_changes,
            "energy_cost": self.energy_cost,
            "time_elapsed": self.time_elapsed,
            "side_effects": self.side_effects,
            "description": self.description,
            "error_message": self.error_message,
        }


class PhysicalActionProcessor:
    """
    Core processor for handling all physical actions in the game world.

    This class coordinates movement, environment interaction, and spatial actions
    by integrating with physics constraints, game state management, and search services.
    """

    def __init__(
        self,
        console: Console,
        game_state_manager: Any = None,
        search_service: Any = None,
        physics_engine: Any = None,
    ):
        """
        Initialize the physical action processor.

        Args:
            console: Rich console for output
            game_state_manager: Game state manager for accessing and updating state
            search_service: Semantic search service for target validation
            physics_engine: Physics constraint engine for action validation
        """
        self.console = console
        self.state_manager = game_state_manager
        self.search_service = search_service
        self.physics = physics_engine
        self._action_handlers: dict[PhysicalActionType, Callable] = {}
        self._constraint_validators: list[Callable] = []
        self._action_metrics: dict[str, Any] = {}
        self._initialize_handlers()

    async def process_physical_action(
        self, action_classification: ActionClassification, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Main entry point for processing physical actions.

        Args:
            action_classification: Classified action from the action classifier
            context: Current game context and state

        Returns:
            PhysicalActionResult containing the outcome of the action
        """
        try:
            # Determine specific physical action type
            physical_action_type = await self._determine_physical_action_type(
                action_classification, context
            )

            # Validate action feasibility
            is_feasible, error_msg = await self.validate_action_feasibility(
                physical_action_type,
                action_classification.secondary_targets or [],
                context,
            )

            if not is_feasible:
                return PhysicalActionResult(
                    success=False,
                    action_type=physical_action_type,
                    affected_entities=[],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Action could not be performed.",
                    error_message=error_msg,
                )

            # Calculate action requirements
            requirements = await self.calculate_action_requirements(
                physical_action_type, action_classification.secondary_targets or []
            )

            # Execute the action based on type
            result = await self._execute_action_by_type(
                physical_action_type, action_classification, context, requirements
            )

            # Apply side effects and update world state
            if result.success:
                await self.update_world_state(result, context)
                side_effects = await self.calculate_side_effects(
                    physical_action_type, result.affected_entities, context
                )
                result.side_effects = side_effects

            # Update metrics
            await self._update_action_metrics(physical_action_type, result)

            return result

        except Exception as e:
            logger.error(f"Error processing physical action: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MOVEMENT,  # Default fallback
                affected_entities=[],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="An error occurred while processing the action.",
                error_message=str(e),
            )

    async def validate_action_feasibility(
        self,
        action_type: PhysicalActionType,
        target_entities: list[str],
        context: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Check if the physical action is feasible in current context.

        Args:
            action_type: Type of physical action to validate
            target_entities: List of entities involved in the action
            context: Current game context

        Returns:
            Tuple of (is_feasible, error_message)
        """
        try:
            # Basic context validation
            if not context.get("player_id"):
                return False, "Player context is required for physical actions."

            player_state = context.get("player_state", {})
            current_location = player_state.get("current_location")

            if not current_location:
                return (
                    False,
                    "Player must be in a valid location to perform physical actions.",
                )

            # Check energy requirements
            if player_state.get("energy", 100) < 10:  # Minimum energy threshold
                return False, "Insufficient energy to perform this action."

            # Run constraint validators if physics engine is available
            if self.physics:
                constraints_valid, constraint_error = (
                    await self.physics.validate_physical_constraints(
                        action_type, target_entities, player_state
                    )
                )
                if not constraints_valid:
                    return False, constraint_error

            # Run custom validators
            for validator in self._constraint_validators:
                is_valid, error_msg = await validator(
                    action_type, target_entities, context
                )
                if not is_valid:
                    return False, error_msg

            return True, None

        except Exception as e:
            logger.error(f"Error validating action feasibility: {e}")
            return False, f"Validation error: {str(e)}"

    async def calculate_action_requirements(
        self, action_type: PhysicalActionType, target_entities: list[str]
    ) -> dict[str, Any]:
        """
        Calculate energy, time, and resource requirements for action.

        Args:
            action_type: Type of physical action
            target_entities: Entities involved in the action

        Returns:
            Dictionary containing action requirements
        """
        # Base requirements by action type
        base_requirements = {
            PhysicalActionType.MOVEMENT: {
                "energy": 5.0,
                "time": 3.0,
                "difficulty": 0.2,
            },
            PhysicalActionType.MANIPULATION: {
                "energy": 10.0,
                "time": 5.0,
                "difficulty": 0.4,
            },
            PhysicalActionType.CLIMBING: {
                "energy": 20.0,
                "time": 10.0,
                "difficulty": 0.7,
            },
            PhysicalActionType.JUMPING: {
                "energy": 15.0,
                "time": 2.0,
                "difficulty": 0.5,
            },
            PhysicalActionType.PUSHING: {
                "energy": 25.0,
                "time": 8.0,
                "difficulty": 0.6,
            },
            PhysicalActionType.PULLING: {
                "energy": 25.0,
                "time": 8.0,
                "difficulty": 0.6,
            },
            PhysicalActionType.OPENING: {"energy": 8.0, "time": 4.0, "difficulty": 0.3},
            PhysicalActionType.CLOSING: {"energy": 8.0, "time": 4.0, "difficulty": 0.3},
            PhysicalActionType.BREAKING: {
                "energy": 30.0,
                "time": 12.0,
                "difficulty": 0.8,
            },
            PhysicalActionType.BUILDING: {
                "energy": 40.0,
                "time": 20.0,
                "difficulty": 0.9,
            },
        }

        requirements = base_requirements.get(
            action_type, {"energy": 10.0, "time": 5.0, "difficulty": 0.5}
        ).copy()

        # Modify requirements based on target entities
        entity_modifier = 1.0 + (
            len(target_entities) * 0.2
        )  # More entities = more work
        requirements["energy"] *= entity_modifier
        requirements["time"] *= entity_modifier

        return requirements

    async def execute_movement_action(
        self, direction: str, distance: float | None, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Execute movement to a new location or direction.

        Args:
            direction: Direction of movement (north, south, east, west, etc.)
            distance: Optional distance to move
            context: Current game context

        Returns:
            PhysicalActionResult with movement outcome
        """
        try:
            player_id = context.get("player_id")
            if not player_id:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=[],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Movement failed.",
                    error_message="Player ID not found in context.",
                )

            # Normalize direction
            normalized_direction = self._normalize_direction(direction)

            # Calculate movement cost
            requirements = await self.calculate_action_requirements(
                PhysicalActionType.MOVEMENT, []
            )

            # For now, simulate movement success
            # In a full implementation, this would integrate with MovementManager
            description = f"You move {normalized_direction}."
            if distance:
                description = f"You move {distance} units {normalized_direction}."

            return PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MOVEMENT,
                affected_entities=[player_id],
                state_changes={"player_location": f"moved_{normalized_direction}"},
                energy_cost=requirements["energy"],
                time_elapsed=requirements["time"],
                side_effects=[],
                description=description,
            )

        except Exception as e:
            logger.error(f"Error executing movement action: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MOVEMENT,
                affected_entities=[],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Movement failed due to an error.",
                error_message=str(e),
            )

    async def execute_manipulation_action(
        self, target_entity: str, action_verb: str, context: dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Execute object manipulation (push, pull, lift, etc.).

        Args:
            target_entity: Entity to manipulate
            action_verb: Specific manipulation action
            context: Current game context

        Returns:
            PhysicalActionResult with manipulation outcome
        """
        try:
            # Determine manipulation type
            manipulation_type = self._get_manipulation_type(action_verb)

            # Calculate requirements
            requirements = await self.calculate_action_requirements(
                manipulation_type, [target_entity]
            )

            # Simulate manipulation
            description = f"You {action_verb} the {target_entity}."

            return PhysicalActionResult(
                success=True,
                action_type=manipulation_type,
                affected_entities=[target_entity],
                state_changes={target_entity: f"{action_verb}_applied"},
                energy_cost=requirements["energy"],
                time_elapsed=requirements["time"],
                side_effects=[],
                description=description,
            )

        except Exception as e:
            logger.error(f"Error executing manipulation action: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Manipulation failed due to an error.",
                error_message=str(e),
            )

    async def execute_environmental_action(
        self,
        target_entity: str,
        action_type: PhysicalActionType,
        context: dict[str, Any],
    ) -> PhysicalActionResult:
        """
        Execute environmental interactions (climbing, jumping, etc.).

        Args:
            target_entity: Environmental entity to interact with
            action_type: Type of environmental action
            context: Current game context

        Returns:
            PhysicalActionResult with environmental action outcome
        """
        try:
            # Calculate requirements
            requirements = await self.calculate_action_requirements(
                action_type, [target_entity]
            )

            # Simulate environmental action
            action_name = action_type.value
            description = f"You {action_name} on the {target_entity}."

            return PhysicalActionResult(
                success=True,
                action_type=action_type,
                affected_entities=[target_entity],
                state_changes={target_entity: f"{action_name}_performed"},
                energy_cost=requirements["energy"],
                time_elapsed=requirements["time"],
                side_effects=[],
                description=description,
            )

        except Exception as e:
            logger.error(f"Error executing environmental action: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=action_type,
                affected_entities=[],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Environmental action failed due to an error.",
                error_message=str(e),
            )

    async def apply_physics_constraints(
        self,
        action_type: PhysicalActionType,
        entities: list[str],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Apply physics constraints and limitations to actions.

        Args:
            action_type: Type of physical action
            entities: Entities involved in the action
            context: Current game context

        Returns:
            List of constraint violations or modifications
        """
        constraints = []

        if self.physics:
            try:
                violations = await self.physics.get_constraint_violations(
                    action_type, context
                )
                constraints.extend(violations)
            except Exception as e:
                logger.error(f"Error applying physics constraints: {e}")

        return constraints

    async def update_world_state(
        self, action_result: PhysicalActionResult, context: dict[str, Any]
    ) -> None:
        """
        Update world state based on successful physical action.

        Args:
            action_result: Result of the physical action
            context: Current game context
        """
        if not self.state_manager:
            return

        try:
            # Update player energy
            player_id = context.get("player_id")
            if player_id and action_result.energy_cost > 0:
                # In a full implementation, this would update player state
                logger.info(
                    f"Player {player_id} expended {action_result.energy_cost} energy"
                )

            # Apply state changes
            for entity, change in action_result.state_changes.items():
                logger.info(f"State change for {entity}: {change}")

        except Exception as e:
            logger.error(f"Error updating world state: {e}")

    async def calculate_side_effects(
        self,
        action_type: PhysicalActionType,
        entities: list[str],
        context: dict[str, Any],
    ) -> list[str]:
        """
        Calculate secondary effects of physical actions.

        Args:
            action_type: Type of physical action
            entities: Entities affected by the action
            context: Current game context

        Returns:
            List of side effect descriptions
        """
        side_effects = []

        # Add some basic side effects based on action type
        if action_type == PhysicalActionType.BREAKING:
            side_effects.append("Debris scattered around the area.")
        elif action_type == PhysicalActionType.JUMPING:
            side_effects.append("You land with a slight thud.")
        elif action_type == PhysicalActionType.CLIMBING:
            side_effects.append("You feel slightly out of breath from the exertion.")

        return side_effects

    def register_action_handler(
        self, action_type: PhysicalActionType, handler: Callable
    ) -> None:
        """
        Register a handler for a specific physical action type.

        Args:
            action_type: Physical action type to handle
            handler: Handler function for the action
        """
        self._action_handlers[action_type] = handler

    def register_constraint_validator(self, validator: Callable) -> None:
        """
        Register a physics constraint validator.

        Args:
            validator: Validator function that checks action constraints
        """
        self._constraint_validators.append(validator)

    async def _determine_physical_action_type(
        self, action_classification: ActionClassification, context: dict[str, Any]
    ) -> PhysicalActionType:
        """Determine the specific physical action type from classification."""
        # Extract action verb or intent to determine physical action type
        intent = action_classification.intent or ""
        target = action_classification.target or ""
        parameters = action_classification.parameters or {}

        # Look for movement indicators
        movement_words = [
            "move",
            "go",
            "walk",
            "run",
            "travel",
            "head",
            "north",
            "south",
            "east",
            "west",
        ]
        if any(word in intent.lower() for word in movement_words):
            return PhysicalActionType.MOVEMENT

        # Look for manipulation indicators
        manipulation_words = ["push", "pull", "lift", "carry", "drag"]
        if any(word in intent.lower() for word in manipulation_words):
            return self._get_manipulation_type(intent)

        # Look for other physical action indicators
        if "climb" in intent.lower():
            return PhysicalActionType.CLIMBING
        elif "jump" in intent.lower():
            return PhysicalActionType.JUMPING
        elif "open" in intent.lower():
            return PhysicalActionType.OPENING
        elif "close" in intent.lower():
            return PhysicalActionType.CLOSING
        elif "break" in intent.lower():
            return PhysicalActionType.BREAKING
        elif "build" in intent.lower():
            return PhysicalActionType.BUILDING

        # Default to manipulation for object interactions
        return PhysicalActionType.MANIPULATION

    def _get_manipulation_type(self, action_verb: str) -> PhysicalActionType:
        """Get manipulation type from action verb."""
        verb_lower = action_verb.lower()

        if "push" in verb_lower:
            return PhysicalActionType.PUSHING
        elif "pull" in verb_lower:
            return PhysicalActionType.PULLING
        else:
            return PhysicalActionType.MANIPULATION

    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction string to standard format."""
        direction_map = {
            "n": "north",
            "north": "north",
            "s": "south",
            "south": "south",
            "e": "east",
            "east": "east",
            "w": "west",
            "west": "west",
            "ne": "northeast",
            "northeast": "northeast",
            "nw": "northwest",
            "northwest": "northwest",
            "se": "southeast",
            "southeast": "southeast",
            "sw": "southwest",
            "southwest": "southwest",
            "u": "up",
            "up": "up",
            "d": "down",
            "down": "down",
        }

        return direction_map.get(direction.lower(), direction.lower())

    async def _execute_action_by_type(
        self,
        action_type: PhysicalActionType,
        action_classification: ActionClassification,
        context: dict[str, Any],
        requirements: dict[str, Any],
    ) -> PhysicalActionResult:
        """Execute action based on its type."""
        target = action_classification.target or ""
        intent = action_classification.intent or ""

        if action_type == PhysicalActionType.MOVEMENT:
            # Extract direction from intent or target
            direction = (
                target
                if target in ["north", "south", "east", "west", "up", "down"]
                else "forward"
            )
            return await self.execute_movement_action(direction, None, context)
        elif action_type in [
            PhysicalActionType.PUSHING,
            PhysicalActionType.PULLING,
            PhysicalActionType.MANIPULATION,
        ]:
            return await self.execute_manipulation_action(target, intent, context)
        else:
            return await self.execute_environmental_action(target, action_type, context)

    async def _update_action_metrics(
        self, action_type: PhysicalActionType, result: PhysicalActionResult
    ) -> None:
        """Update action metrics for monitoring and optimization."""
        metric_key = action_type.value
        if metric_key not in self._action_metrics:
            self._action_metrics[metric_key] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "total_energy_cost": 0.0,
                "total_time": 0.0,
            }

        metrics = self._action_metrics[metric_key]
        metrics["total_attempts"] += 1
        if result.success:
            metrics["successful_attempts"] += 1
        metrics["total_energy_cost"] += result.energy_cost
        metrics["total_time"] += result.time_elapsed

    async def _validate_entity_accessibility(
        self, entity_id: str, context: dict[str, Any]
    ) -> bool:
        """Check if entity is accessible for physical interaction."""
        # In a full implementation, this would check distance, obstacles, etc.
        return True

    async def _calculate_energy_cost(
        self, action_type: PhysicalActionType, difficulty: float
    ) -> float:
        """Calculate energy cost for physical action."""
        base_costs = {
            PhysicalActionType.MOVEMENT: 5.0,
            PhysicalActionType.MANIPULATION: 10.0,
            PhysicalActionType.CLIMBING: 20.0,
            PhysicalActionType.JUMPING: 15.0,
            PhysicalActionType.PUSHING: 25.0,
            PhysicalActionType.PULLING: 25.0,
            PhysicalActionType.OPENING: 8.0,
            PhysicalActionType.CLOSING: 8.0,
            PhysicalActionType.BREAKING: 30.0,
            PhysicalActionType.BUILDING: 40.0,
        }

        base_cost = base_costs.get(action_type, 10.0)
        return base_cost * (1.0 + difficulty)

    def _initialize_handlers(self) -> None:
        """Initialize default action handlers."""
        # Register default handlers for each action type
        self._action_handlers[PhysicalActionType.MOVEMENT] = (
            self.execute_movement_action
        )
        self._action_handlers[PhysicalActionType.MANIPULATION] = (
            self.execute_manipulation_action
        )
        # Additional handlers would be registered here in a full implementation

    def get_action_metrics(self) -> dict[str, Any]:
        """Get action performance metrics."""
        return self._action_metrics.copy()
