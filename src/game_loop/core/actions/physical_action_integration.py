"""
Physical Action Integration layer for coordinating physical action processing.

This module integrates physical action processing with the main game systems,
providing coordination between all physical action components.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from game_loop.core.actions.types import ActionClassification
from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionProcessor,
    PhysicalActionResult,
    PhysicalActionType,
)
from game_loop.core.environment.interaction_manager import EnvironmentInteractionManager
from game_loop.core.movement.movement_manager import MovementManager
from game_loop.core.navigation.spatial_navigator import SpatialNavigator
from game_loop.core.physics.constraint_engine import PhysicsConstraintEngine

logger = logging.getLogger(__name__)


class ActionExecutionContext:
    """Context for action execution with all necessary components."""

    def __init__(
        self,
        physical_processor: PhysicalActionProcessor,
        movement_manager: MovementManager,
        environment_manager: EnvironmentInteractionManager,
        physics_engine: PhysicsConstraintEngine,
        spatial_navigator: SpatialNavigator,
        game_state_manager: Any,
    ):
        self.physical_processor = physical_processor
        self.movement = movement_manager
        self.environment = environment_manager
        self.physics = physics_engine
        self.navigator = spatial_navigator
        self.state = game_state_manager


class PhysicalActionIntegration:
    """
    Integrate physical action processing with the main game systems.

    This class coordinates execution of physical actions by managing the interaction
    between all physical action components and the broader game systems.
    """

    def __init__(
        self,
        physical_processor: PhysicalActionProcessor,
        movement_manager: MovementManager,
        environment_manager: EnvironmentInteractionManager,
        physics_engine: PhysicsConstraintEngine,
        spatial_navigator: SpatialNavigator,
        game_state_manager: Any,
    ):
        """
        Initialize the physical action integration.

        Args:
            physical_processor: Main physical action processor
            movement_manager: Movement and navigation manager
            environment_manager: Environment interaction manager
            physics_engine: Physics constraint engine
            spatial_navigator: Spatial navigation system
            game_state_manager: Game state manager
        """
        self.execution_context = ActionExecutionContext(
            physical_processor,
            movement_manager,
            environment_manager,
            physics_engine,
            spatial_navigator,
            game_state_manager,
        )
        self._action_metrics: dict[str, Any] = {}
        self._integration_hooks: dict[str, list[Callable]] = {
            "pre_execution": [],
            "post_execution": [],
            "error_handling": [],
            "state_update": [],
        }
        self._active_actions: dict[str, dict[str, Any]] = {}
        self._action_history: list[dict[str, Any]] = []

    async def process_classified_physical_action(
        self, action_classification: ActionClassification, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Main entry point for processing classified physical actions.

        Args:
            action_classification: Classified action from action classifier
            context: Current game context and state

        Returns:
            Dictionary containing action result and metadata
        """
        try:
            action_id = f"action_{asyncio.get_event_loop().time()}"

            # Record action start
            self._active_actions[action_id] = {
                "classification": action_classification,
                "context": context,
                "start_time": asyncio.get_event_loop().time(),
                "status": "processing",
            }

            # Execute pre-processing hooks
            enhanced_context = await self._execute_pre_execution_hooks(
                action_classification, context
            )

            # Validate action feasibility with enhanced context
            is_feasible, feasibility_error = (
                await self.validate_action_chain_feasibility(
                    [
                        {
                            "classification": action_classification,
                            "context": enhanced_context,
                        }
                    ],
                    enhanced_context,
                )
            )

            if not is_feasible:
                result = {
                    "success": False,
                    "error": feasibility_error,
                    "action_type": "physical",
                    "classification": action_classification.to_dict(),
                }
                await self._handle_action_completion(action_id, result)
                return result

            # Determine the appropriate execution strategy
            execution_strategy = await self._determine_execution_strategy(
                action_classification, enhanced_context
            )

            # Execute the action based on strategy
            if execution_strategy == "movement":
                result = await self._execute_movement_action(
                    action_classification, enhanced_context
                )
            elif execution_strategy == "environment":
                result = await self._execute_environment_action(
                    action_classification, enhanced_context
                )
            elif execution_strategy == "complex":
                result = await self._execute_complex_action(
                    action_classification, enhanced_context
                )
            else:
                # Default to physical processor
                action_result = await self.execution_context.physical_processor.process_physical_action(
                    action_classification, enhanced_context
                )
                result = await self._convert_action_result_to_dict(action_result)

            # Apply learning effects
            if result.get("success", False):
                await self.apply_learning_effects(
                    enhanced_context.get("player_id", "unknown"),
                    action_classification.action_type,
                    1.0,  # Success rate
                )

            # Execute post-processing hooks
            final_result = await self._execute_post_execution_hooks(
                result, action_classification
            )

            # Add classification to result
            final_result["classification"] = action_classification.to_dict()

            # Update integration metrics
            await self._update_integration_metrics(action_classification, final_result)

            # Record action completion
            await self._handle_action_completion(action_id, final_result)

            return final_result

        except Exception as e:
            logger.error(f"Error processing classified physical action: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "action_type": "physical",
                "classification": action_classification.to_dict(),
            }

            # Execute error handling hooks
            await self._execute_error_hooks(e, action_classification, context)

            return error_result

    async def coordinate_multi_step_actions(
        self, action_sequence: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[PhysicalActionResult]:
        """
        Coordinate execution of multi-step physical actions.

        Args:
            action_sequence: Sequence of actions to execute
            context: Current game context

        Returns:
            List of PhysicalActionResult for each step
        """
        try:
            results = []
            updated_context = context.copy()

            for i, action_step in enumerate(action_sequence):
                step_classification = action_step.get("classification")
                step_context = action_step.get("context", updated_context)

                logger.info(f"Executing action step {i + 1}/{len(action_sequence)}")

                # Check if previous steps affect this step's feasibility
                if i > 0:
                    step_context = await self._update_context_from_previous_results(
                        step_context, results
                    )

                # Execute the step
                step_result_dict = await self.process_classified_physical_action(
                    step_classification, step_context
                )

                # Convert to PhysicalActionResult if needed
                if isinstance(step_result_dict, dict):
                    step_result = await self._convert_dict_to_action_result(
                        step_result_dict
                    )
                else:
                    step_result = step_result_dict

                results.append(step_result)

                # Update context for next step
                if step_result.success:
                    updated_context.update(step_result.state_changes)
                else:
                    # Handle step failure
                    failure_strategy = await self._handle_multi_step_failure(
                        i, action_sequence, results, updated_context
                    )

                    if failure_strategy == "abort":
                        break
                    elif failure_strategy == "retry":
                        # Retry the current step once
                        retry_result_dict = (
                            await self.process_classified_physical_action(
                                step_classification, step_context
                            )
                        )
                        retry_result = await self._convert_dict_to_action_result(
                            retry_result_dict
                        )
                        results[-1] = retry_result

                        if retry_result.success:
                            updated_context.update(retry_result.state_changes)
                        else:
                            break

                # Add delay between steps if needed
                if i < len(action_sequence) - 1:
                    await asyncio.sleep(0.1)  # Small delay between steps

            return results

        except Exception as e:
            logger.error(f"Error coordinating multi-step actions: {e}")
            return []

    async def handle_action_interruptions(
        self, active_action: dict[str, Any], interruption_event: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle interruptions during physical action execution.

        Args:
            active_action: Currently executing action
            interruption_event: Event that interrupts the action

        Returns:
            Dictionary containing interruption handling result
        """
        try:
            interruption_type = interruption_event.get("type", "unknown")
            interruption_severity = interruption_event.get("severity", "medium")

            # Determine if action can continue
            can_continue = await self._can_action_continue_after_interruption(
                active_action, interruption_event
            )

            if can_continue:
                # Apply interruption effects but continue action
                effects = await self._apply_interruption_effects(
                    active_action, interruption_event
                )

                return {
                    "action_status": "continued",
                    "effects": effects,
                    "modified_action": active_action,
                    "interruption_handled": True,
                }
            else:
                # Action must be stopped
                partial_result = await self._calculate_partial_action_result(
                    active_action, interruption_event
                )

                return {
                    "action_status": "interrupted",
                    "partial_result": partial_result,
                    "interruption_cause": interruption_type,
                    "recovery_options": await self._generate_recovery_options(
                        active_action, interruption_event
                    ),
                }

        except Exception as e:
            logger.error(f"Error handling action interruption: {e}")
            return {
                "action_status": "error",
                "error": str(e),
                "interruption_handled": False,
            }

    async def optimize_action_performance(
        self, action_type: PhysicalActionType, frequency_data: dict[str, Any]
    ) -> None:
        """
        Optimize performance for frequently used actions.

        Args:
            action_type: Type of physical action to optimize
            frequency_data: Usage frequency and performance data
        """
        try:
            optimization_strategies = []

            # Analyze frequency data
            usage_count = frequency_data.get("usage_count", 0)
            avg_execution_time = frequency_data.get("avg_execution_time", 0.0)
            success_rate = frequency_data.get("success_rate", 1.0)

            # Suggest optimizations based on usage patterns
            if usage_count > 100:  # Frequently used action
                optimization_strategies.append("cache_validation_results")
                optimization_strategies.append("precompute_constraints")

            if avg_execution_time > 5.0:  # Slow action
                optimization_strategies.append("parallel_processing")
                optimization_strategies.append("optimize_pathfinding")

            if success_rate < 0.8:  # Low success rate
                optimization_strategies.append("improve_error_detection")
                optimization_strategies.append("enhance_feasibility_checking")

            # Apply optimizations
            for strategy in optimization_strategies:
                await self._apply_optimization_strategy(action_type, strategy)

            logger.info(
                f"Applied {len(optimization_strategies)} optimizations for {action_type.value}"
            )

        except Exception as e:
            logger.error(f"Error optimizing action performance: {e}")

    async def validate_action_chain_feasibility(
        self, action_chain: list[dict[str, Any]], context: dict[str, Any]
    ) -> tuple[bool, int | None]:
        """
        Validate that a chain of actions can be completed.

        Args:
            action_chain: List of actions to validate
            context: Current game context

        Returns:
            Tuple of (is_feasible, first_failing_step_index)
        """
        try:
            simulated_context = context.copy()

            for i, action_step in enumerate(action_chain):
                classification = action_step.get("classification")
                step_context = action_step.get("context", simulated_context)

                # Validate individual step
                is_feasible, error_msg = (
                    await self.execution_context.physical_processor.validate_action_feasibility(
                        await self._extract_physical_action_type(classification),
                        classification.secondary_targets or [],
                        step_context,
                    )
                )

                if not is_feasible:
                    logger.info(f"Action chain fails at step {i}: {error_msg}")
                    return False, i

                # Simulate step execution to update context for next step
                simulated_result = await self._simulate_action_execution(
                    classification, step_context
                )
                simulated_context.update(simulated_result.get("state_changes", {}))

            return True, None

        except Exception as e:
            logger.error(f"Error validating action chain feasibility: {e}")
            return False, 0

    async def apply_learning_effects(
        self, player_id: str, action_type: PhysicalActionType, success_rate: float
    ) -> None:
        """
        Apply skill learning effects from repeated actions.

        Args:
            player_id: ID of the player
            action_type: Type of physical action
            success_rate: Success rate for the action type
        """
        try:
            if player_id not in self._action_metrics:
                self._action_metrics[player_id] = {}

            player_metrics = self._action_metrics[player_id]
            action_key = action_type.value

            if action_key not in player_metrics:
                player_metrics[action_key] = {
                    "attempts": 0,
                    "successes": 0,
                    "skill_level": 1.0,
                    "experience": 0.0,
                }

            action_data = player_metrics[action_key]
            action_data["attempts"] += 1

            if success_rate > 0.5:  # Consider it a success
                action_data["successes"] += 1
                experience_gain = success_rate * 10.0  # Base experience
                action_data["experience"] += experience_gain

                # Skill level improvement
                if action_data["experience"] > action_data["skill_level"] * 100:
                    action_data["skill_level"] += 0.1
                    action_data["experience"] = 0.0
                    logger.info(
                        f"Player {player_id} improved {action_type.value} skill to {action_data['skill_level']:.1f}"
                    )

        except Exception as e:
            logger.error(f"Error applying learning effects: {e}")

    async def handle_concurrent_actions(
        self, actions: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[PhysicalActionResult]:
        """
        Handle multiple physical actions happening simultaneously.

        Args:
            actions: List of actions to execute concurrently
            context: Current game context

        Returns:
            List of PhysicalActionResult for each action
        """
        try:
            # Check for action conflicts
            conflicts = await self._detect_action_conflicts(actions, context)

            if conflicts:
                # Resolve conflicts by prioritizing actions
                resolved_actions = await self._resolve_action_conflicts(
                    actions, conflicts
                )
            else:
                resolved_actions = actions

            # Execute actions concurrently
            tasks = []
            for action in resolved_actions:
                classification = action.get("classification")
                action_context = action.get("context", context)

                task = asyncio.create_task(
                    self.process_classified_physical_action(
                        classification, action_context
                    )
                )
                tasks.append(task)

            # Wait for all actions to complete
            results_dicts = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert results and handle exceptions
            results = []
            for result in results_dicts:
                if isinstance(result, Exception):
                    error_result = PhysicalActionResult(
                        success=False,
                        action_type=PhysicalActionType.MANIPULATION,
                        affected_entities=[],
                        state_changes={},
                        energy_cost=0.0,
                        time_elapsed=0.0,
                        side_effects=[],
                        description="Concurrent action failed.",
                        error_message=str(result),
                    )
                    results.append(error_result)
                else:
                    action_result = await self._convert_dict_to_action_result(result)
                    results.append(action_result)

            return results

        except Exception as e:
            logger.error(f"Error handling concurrent actions: {e}")
            return []

    async def generate_action_feedback(
        self, action_result: PhysicalActionResult, context: dict[str, Any]
    ) -> str:
        """
        Generate descriptive feedback for physical action results.

        Args:
            action_result: Result of the physical action
            context: Current game context

        Returns:
            Human-readable description of the action result
        """
        try:
            base_description = action_result.description

            # Enhance description based on context and results
            enhancements = []

            # Add energy/time information if significant
            if action_result.energy_cost > 20:
                enhancements.append(
                    f"The effort leaves you feeling tired ({action_result.energy_cost:.1f} energy used)."
                )

            if action_result.time_elapsed > 10:
                enhancements.append(
                    f"The action took {action_result.time_elapsed:.1f} seconds to complete."
                )

            # Add side effects
            for side_effect in action_result.side_effects:
                enhancements.append(side_effect)

            # Add physics-based feedback
            if action_result.action_type in [
                PhysicalActionType.PUSHING,
                PhysicalActionType.PULLING,
            ]:
                enhancements.append(
                    "You feel the resistance of the object against your effort."
                )
            elif action_result.action_type == PhysicalActionType.CLIMBING:
                enhancements.append("Your muscles strain as you pull yourself upward.")

            # Combine base description with enhancements
            if enhancements:
                enhanced_description = f"{base_description}\n\n{' '.join(enhancements)}"
            else:
                enhanced_description = base_description

            return enhanced_description

        except Exception as e:
            logger.error(f"Error generating action feedback: {e}")
            return action_result.description

    async def update_world_physics_state(
        self, action_results: list[PhysicalActionResult]
    ) -> None:
        """
        Update world physics state based on action results.

        Args:
            action_results: List of action results to process
        """
        try:
            for result in action_results:
                if result.success:
                    # Update physics state for each affected entity
                    for entity in result.affected_entities:
                        await self._update_entity_physics_state(entity, result)

                    # Apply global physics effects
                    if result.action_type == PhysicalActionType.BREAKING:
                        await self._handle_destruction_physics(result)
                    elif result.action_type in [
                        PhysicalActionType.PUSHING,
                        PhysicalActionType.PULLING,
                    ]:
                        await self._handle_movement_physics(result)

        except Exception as e:
            logger.error(f"Error updating world physics state: {e}")

    def register_action_integration_hook(
        self, hook_type: str, handler: Callable
    ) -> None:
        """
        Register integration hooks for action processing.

        Args:
            hook_type: Type of hook (pre_execution, post_execution, etc.)
            handler: Handler function for the hook
        """
        if hook_type in self._integration_hooks:
            self._integration_hooks[hook_type].append(handler)
        else:
            logger.warning(f"Unknown hook type: {hook_type}")

    # Private helper methods

    async def _execute_pre_execution_hooks(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute pre-execution hooks."""
        enhanced_context = context.copy()

        for hook in self._integration_hooks["pre_execution"]:
            try:
                hook_result = await hook(classification, enhanced_context)
                if isinstance(hook_result, dict):
                    enhanced_context.update(hook_result)
            except Exception as e:
                logger.error(f"Error in pre-execution hook: {e}")

        return enhanced_context

    async def _execute_post_execution_hooks(
        self, result: dict[str, Any], classification: ActionClassification
    ) -> dict[str, Any]:
        """Execute post-execution hooks."""
        enhanced_result = result.copy()

        for hook in self._integration_hooks["post_execution"]:
            try:
                hook_result = await hook(enhanced_result, classification)
                if isinstance(hook_result, dict):
                    enhanced_result.update(hook_result)
            except Exception as e:
                logger.error(f"Error in post-execution hook: {e}")

        return enhanced_result

    async def _execute_error_hooks(
        self,
        error: Exception,
        classification: ActionClassification,
        context: dict[str, Any],
    ) -> None:
        """Execute error handling hooks."""
        for hook in self._integration_hooks["error_handling"]:
            try:
                await hook(error, classification, context)
            except Exception as e:
                logger.error(f"Error in error handling hook: {e}")

    async def _determine_execution_strategy(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> str:
        """Determine the appropriate execution strategy."""
        if classification.action_type.value in [
            "movement",
            "physical",
        ] and classification.target in ["north", "south", "east", "west"]:
            return "movement"
        elif classification.intent and any(
            word in classification.intent.lower()
            for word in ["container", "open", "close", "use"]
        ):
            return "environment"
        elif len(classification.secondary_targets or []) > 2:
            return "complex"
        else:
            return "default"

    async def _execute_movement_action(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute movement-specific action."""
        try:
            player_id = context.get("player_id", "unknown")
            direction = classification.target or classification.intent or "forward"

            result = await self.execution_context.movement.process_movement_command(
                player_id, direction, context
            )

            return await self._convert_action_result_to_dict(result)
        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "movement"}

    async def _execute_environment_action(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute environment-specific action."""
        try:
            player_id = context.get("player_id", "unknown")
            target = classification.target or "unknown"
            intent = classification.intent or "interact"

            result = await self.execution_context.environment.process_environment_interaction(
                player_id, target, intent, context
            )

            return await self._convert_action_result_to_dict(result)
        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "environment"}

    async def _execute_complex_action(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute complex multi-component action."""
        try:
            # Break down complex action into simpler components
            components = await self._decompose_complex_action(classification, context)

            if len(components) > 1:
                # Execute as multi-step action
                results = await self.coordinate_multi_step_actions(components, context)

                # Combine results
                combined_result = await self._combine_action_results(results)
                return await self._convert_action_result_to_dict(combined_result)
            else:
                # Execute as single action
                action_result = await self.execution_context.physical_processor.process_physical_action(
                    classification, context
                )
                return await self._convert_action_result_to_dict(action_result)

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "complex"}

    async def _convert_action_result_to_dict(
        self, result: PhysicalActionResult
    ) -> dict[str, Any]:
        """Convert PhysicalActionResult to dictionary."""
        return result.to_dict()

    async def _convert_dict_to_action_result(
        self, result_dict: dict[str, Any]
    ) -> PhysicalActionResult:
        """Convert dictionary to PhysicalActionResult."""
        action_type_value = result_dict.get("action_type")
        if isinstance(action_type_value, str):
            # If it's a string like "physical", default to manipulation
            # In a full implementation, this would properly map action types
            if action_type_value in ["physical", "unknown"]:
                action_type = PhysicalActionType.MANIPULATION
            else:
                try:
                    action_type = PhysicalActionType(action_type_value)
                except ValueError:
                    action_type = PhysicalActionType.MANIPULATION
        else:
            action_type = (
                action_type_value
                if action_type_value
                else PhysicalActionType.MANIPULATION
            )

        return PhysicalActionResult(
            success=result_dict.get("success", False),
            action_type=action_type,
            affected_entities=result_dict.get("affected_entities", []),
            state_changes=result_dict.get("state_changes", {}),
            energy_cost=result_dict.get("energy_cost", 0.0),
            time_elapsed=result_dict.get("time_elapsed", 0.0),
            side_effects=result_dict.get("side_effects", []),
            description=result_dict.get("description", "Action completed."),
            error_message=result_dict.get("error_message"),
        )

    async def _extract_physical_action_type(
        self, classification: ActionClassification
    ) -> PhysicalActionType:
        """Extract physical action type from classification."""
        intent = classification.intent or ""

        if "move" in intent.lower() or classification.target in [
            "north",
            "south",
            "east",
            "west",
        ]:
            return PhysicalActionType.MOVEMENT
        elif "push" in intent.lower():
            return PhysicalActionType.PUSHING
        elif "pull" in intent.lower():
            return PhysicalActionType.PULLING
        else:
            return PhysicalActionType.MANIPULATION

    async def _simulate_action_execution(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate action execution for planning purposes."""
        return {
            "state_changes": {"simulated": True},
            "energy_cost": 5.0,
            "time_elapsed": 3.0,
        }

    async def _handle_action_completion(
        self, action_id: str, result: dict[str, Any]
    ) -> None:
        """Handle action completion cleanup and recording."""
        if action_id in self._active_actions:
            action_data = self._active_actions[action_id]
            action_data["status"] = "completed"
            action_data["result"] = result
            action_data["end_time"] = asyncio.get_event_loop().time()

            # Move to history
            self._action_history.append(action_data)
            del self._active_actions[action_id]

            # Limit history size
            if len(self._action_history) > 1000:
                self._action_history = self._action_history[-500:]

    async def _update_integration_metrics(
        self, classification: ActionClassification, result: dict[str, Any]
    ) -> None:
        """Update integration performance metrics."""
        action_type = classification.action_type.value

        if action_type not in self._action_metrics:
            self._action_metrics[action_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "avg_execution_time": 0.0,
                "total_execution_time": 0.0,
            }

        metrics = self._action_metrics[action_type]
        metrics["total_executions"] += 1

        if result.get("success", False):
            metrics["successful_executions"] += 1

        execution_time = result.get("time_elapsed", 0.0)
        metrics["total_execution_time"] += execution_time
        metrics["avg_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_executions"]
        )

    # Additional placeholder methods for complex functionality

    async def _validate_resource_availability(
        self, required_resources: dict[str, Any], player_state: dict[str, Any]
    ) -> bool:
        return True

    async def _calculate_composite_action_cost(
        self, actions: list[dict[str, Any]]
    ) -> dict[str, float]:
        return {"energy": 10.0, "time": 5.0}

    def _should_cache_action_result(
        self, action_type: PhysicalActionType, context: dict[str, Any]
    ) -> bool:
        return action_type in [
            PhysicalActionType.MOVEMENT,
            PhysicalActionType.MANIPULATION,
        ]

    async def _update_context_from_previous_results(
        self, context: dict[str, Any], results: list[PhysicalActionResult]
    ) -> dict[str, Any]:
        updated_context = context.copy()
        for result in results:
            updated_context.update(result.state_changes)
        return updated_context

    async def _handle_multi_step_failure(
        self,
        step_index: int,
        action_sequence: list[dict[str, Any]],
        results: list[PhysicalActionResult],
        context: dict[str, Any],
    ) -> str:
        return "abort"  # Simple strategy: abort on failure

    async def _can_action_continue_after_interruption(
        self, active_action: dict[str, Any], interruption_event: dict[str, Any]
    ) -> bool:
        return interruption_event.get("severity", "medium") != "critical"

    async def _apply_interruption_effects(
        self, active_action: dict[str, Any], interruption_event: dict[str, Any]
    ) -> dict[str, Any]:
        return {"energy_penalty": 5.0, "time_penalty": 2.0}

    async def _calculate_partial_action_result(
        self, active_action: dict[str, Any], interruption_event: dict[str, Any]
    ) -> dict[str, Any]:
        return {"partial_completion": 0.5, "effects": ["action_interrupted"]}

    async def _generate_recovery_options(
        self, active_action: dict[str, Any], interruption_event: dict[str, Any]
    ) -> list[str]:
        return ["retry_action", "modify_approach", "abandon_action"]

    async def _apply_optimization_strategy(
        self, action_type: PhysicalActionType, strategy: str
    ) -> None:
        logger.info(
            f"Applied optimization strategy '{strategy}' for {action_type.value}"
        )

    async def _detect_action_conflicts(
        self, actions: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return []  # No conflicts detected in simplified implementation

    async def _resolve_action_conflicts(
        self, actions: list[dict[str, Any]], conflicts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return actions  # Return actions unchanged in simplified implementation

    async def _decompose_complex_action(
        self, classification: ActionClassification, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Simple decomposition: return single component
        return [{"classification": classification, "context": context}]

    async def _combine_action_results(
        self, results: list[PhysicalActionResult]
    ) -> PhysicalActionResult:
        if not results:
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MANIPULATION,
                affected_entities=[],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="No results to combine.",
            )

        # Combine multiple results into one
        combined_success = all(r.success for r in results)
        combined_entities = []
        combined_changes = {}
        total_energy = 0.0
        total_time = 0.0
        combined_effects = []

        for result in results:
            combined_entities.extend(result.affected_entities)
            combined_changes.update(result.state_changes)
            total_energy += result.energy_cost
            total_time += result.time_elapsed
            combined_effects.extend(result.side_effects)

        return PhysicalActionResult(
            success=combined_success,
            action_type=results[0].action_type,
            affected_entities=list(set(combined_entities)),
            state_changes=combined_changes,
            energy_cost=total_energy,
            time_elapsed=total_time,
            side_effects=combined_effects,
            description=f"Combined action affecting {len(combined_entities)} entities.",
        )

    async def _update_entity_physics_state(
        self, entity: str, result: PhysicalActionResult
    ) -> None:
        logger.info(f"Updated physics state for entity {entity}")

    async def _handle_destruction_physics(self, result: PhysicalActionResult) -> None:
        logger.info("Applied destruction physics effects")

    async def _handle_movement_physics(self, result: PhysicalActionResult) -> None:
        logger.info("Applied movement physics effects")
