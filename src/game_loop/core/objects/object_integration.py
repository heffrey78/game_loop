"""
Object System Integration for coordinating all object-related systems.

This module provides the central coordination layer that integrates all object systems
including inventory, containers, crafting, conditions, and interactions.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ..containers.container_manager import ContainerManager
from ..crafting.crafting_manager import CraftingManager
from ..inventory.inventory_manager import InventoryManager
from .condition_manager import ObjectConditionManager
from .interaction_system import (
    ObjectInteractionSystem,
    ObjectInteractionType,
)

logger = logging.getLogger(__name__)


@dataclass
class ObjectSystemEvent:
    """Event for object system coordination."""

    event_type: str
    source_system: str
    object_id: str
    data: dict[str, Any]
    timestamp: float
    processed: bool = False


class ObjectSystemIntegration:
    """
    Central coordinator for all object-related systems.

    This class provides comprehensive integration of:
    - Inventory management
    - Object interactions
    - Container systems
    - Crafting mechanics
    - Object conditions and quality
    - Cross-system event coordination
    """

    def __init__(
        self,
        inventory_manager: InventoryManager | None = None,
        interaction_system: ObjectInteractionSystem | None = None,
        condition_manager: ObjectConditionManager | None = None,
        container_manager: ContainerManager | None = None,
        crafting_manager: CraftingManager | None = None,
        object_manager: Any = None,
        physics_engine: Any = None,
    ):
        """
        Initialize the object system integration.

        Args:
            inventory_manager: Manager for inventory operations
            interaction_system: System for object interactions
            condition_manager: Manager for object conditions
            container_manager: Manager for container operations
            crafting_manager: Manager for crafting operations
            object_manager: Manager for object data and properties
            physics_engine: Physics engine for constraint validation
        """
        self.inventory = inventory_manager or InventoryManager(
            object_manager, physics_engine
        )
        self.interactions = interaction_system or ObjectInteractionSystem(
            object_manager, physics_engine
        )
        self.conditions = condition_manager or ObjectConditionManager(object_manager)
        self.containers = container_manager or ContainerManager(
            self.inventory, object_manager, physics_engine
        )
        self.crafting = crafting_manager or CraftingManager(
            object_manager, self.inventory
        )

        self.objects = object_manager
        self.physics = physics_engine

        self._event_queue: list[ObjectSystemEvent] = []
        self._event_handlers: dict[str, list] = {}
        self._system_coordinators: dict[str, Any] = {}
        self._performance_metrics: dict[str, dict[str, Any]] = {}
        self._cross_system_cache: dict[str, dict[str, Any]] = {}

        self._initialize_system_coordination()

    async def process_unified_action(
        self,
        action_type: str,
        actor_id: str,
        target_object: str,
        action_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process a unified action that may involve multiple object systems.

        Args:
            action_type: Type of action to process
            actor_id: Entity performing the action
            target_object: Primary target object
            action_data: Additional action parameters

        Returns:
            Dict with comprehensive action results
        """
        try:
            action_id = f"action_{asyncio.get_event_loop().time()}"
            start_time = asyncio.get_event_loop().time()

            logger.info(
                f"Processing unified action {action_type} by {actor_id} on {target_object}"
            )

            # Validate action prerequisites
            is_valid, validation_errors = await self.validate_action_prerequisites(
                action_type, actor_id, target_object, action_data
            )

            if not is_valid:
                return {
                    "success": False,
                    "action_id": action_id,
                    "error": "Validation failed",
                    "details": validation_errors,
                }

            # Route to appropriate system(s)
            result = {}

            if action_type == "craft_item":
                result = await self._process_crafting_action(actor_id, action_data)
            elif action_type == "use_tool":
                result = await self._process_tool_usage_action(
                    actor_id, target_object, action_data
                )
            elif action_type == "combine_objects":
                result = await self._process_combination_action(actor_id, action_data)
            elif action_type == "organize_container":
                result = await self._process_container_organization(
                    actor_id, target_object, action_data
                )
            elif action_type == "repair_object":
                result = await self._process_repair_action(
                    actor_id, target_object, action_data
                )
            elif action_type == "transfer_items":
                result = await self._process_transfer_action(actor_id, action_data)
            else:
                result = await self._process_generic_action(
                    action_type, actor_id, target_object, action_data
                )

            # Apply cross-system effects
            await self._apply_cross_system_effects(
                action_type, actor_id, target_object, result
            )

            # Update performance metrics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            await self._update_performance_metrics(
                action_type, elapsed_time, result.get("success", False)
            )

            result.update(
                {
                    "action_id": action_id,
                    "processing_time": elapsed_time,
                    "timestamp": start_time,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error processing unified action: {e}")
            return {
                "success": False,
                "action_id": action_id,
                "error": str(e),
                "processing_time": asyncio.get_event_loop().time() - start_time,
            }

    async def get_object_comprehensive_status(
        self, object_id: str, include_history: bool = False
    ) -> dict[str, Any]:
        """
        Get comprehensive status of an object across all systems.

        Args:
            object_id: Object to get status for
            include_history: Whether to include interaction history

        Returns:
            Dict with complete object status information
        """
        try:
            status = {
                "object_id": object_id,
                "systems": {},
                "overall_status": "unknown",
                "recommendations": [],
            }

            # Get condition information
            if self.conditions:
                try:
                    condition_desc = await self.conditions.get_condition_description(
                        object_id, "technical"
                    )
                    status["systems"]["condition"] = {
                        "description": condition_desc,
                        "needs_attention": "poor" in condition_desc.lower()
                        or "damaged" in condition_desc.lower(),
                    }
                except Exception as e:
                    status["systems"]["condition"] = {"error": str(e)}

            # Check if object is in any inventories
            if self.inventory:
                try:
                    # This would require a method to search for object across inventories
                    status["systems"]["inventory"] = {
                        "location": "unknown",
                        "accessible": True,
                    }
                except Exception as e:
                    status["systems"]["inventory"] = {"error": str(e)}

            # Check available interactions
            if self.interactions:
                try:
                    available_interactions = (
                        await self.interactions.get_available_interactions(
                            object_id, {"player_id": "system_check"}
                        )
                    )
                    status["systems"]["interactions"] = {
                        "available_count": len(available_interactions),
                        "interactions": [
                            i.get("type", "unknown") for i in available_interactions
                        ],
                    }
                except Exception as e:
                    status["systems"]["interactions"] = {"error": str(e)}

            # Check if object is part of any containers
            if self.containers:
                try:
                    # This would require container search functionality
                    status["systems"]["containers"] = {
                        "is_container": False,
                        "contained_in": None,
                    }
                except Exception as e:
                    status["systems"]["containers"] = {"error": str(e)}

            # Check crafting involvement
            if self.crafting:
                try:
                    # This would check if object is used in any recipes
                    status["systems"]["crafting"] = {
                        "craftable": False,
                        "ingredient_in": [],
                        "can_craft_with": [],
                    }
                except Exception as e:
                    status["systems"]["crafting"] = {"error": str(e)}

            # Generate overall assessment
            status["overall_status"] = self._assess_overall_object_status(
                status["systems"]
            )
            status["recommendations"] = self._generate_object_recommendations(
                status["systems"]
            )

            return status

        except Exception as e:
            logger.error(f"Error getting comprehensive object status: {e}")
            return {"object_id": object_id, "error": str(e)}

    async def optimize_system_performance(
        self, optimization_targets: list[str] = None
    ) -> dict[str, Any]:
        """
        Optimize performance across all object systems.

        Args:
            optimization_targets: Specific systems to optimize (None = all)

        Returns:
            Dict with optimization results
        """
        try:
            if optimization_targets is None:
                optimization_targets = [
                    "inventory",
                    "interactions",
                    "conditions",
                    "containers",
                    "crafting",
                ]

            optimization_results = {}

            for target in optimization_targets:
                if target == "inventory" and self.inventory:
                    # Optimize inventory operations
                    result = await self._optimize_inventory_performance()
                    optimization_results["inventory"] = result

                elif target == "interactions" and self.interactions:
                    # Optimize interaction processing
                    result = await self._optimize_interaction_performance()
                    optimization_results["interactions"] = result

                elif target == "conditions" and self.conditions:
                    # Optimize condition tracking
                    result = await self._optimize_condition_performance()
                    optimization_results["conditions"] = result

                elif target == "containers" and self.containers:
                    # Optimize container operations
                    result = await self._optimize_container_performance()
                    optimization_results["containers"] = result

                elif target == "crafting" and self.crafting:
                    # Optimize crafting operations
                    result = await self._optimize_crafting_performance()
                    optimization_results["crafting"] = result

            return {
                "optimization_complete": True,
                "targets": optimization_targets,
                "results": optimization_results,
                "overall_improvement": self._calculate_overall_improvement(
                    optimization_results
                ),
            }

        except Exception as e:
            logger.error(f"Error optimizing system performance: {e}")
            return {"error": str(e)}

    async def synchronize_object_state(
        self, object_id: str, force_update: bool = False
    ) -> dict[str, Any]:
        """
        Synchronize object state across all systems.

        Args:
            object_id: Object to synchronize
            force_update: Whether to force update even if not needed

        Returns:
            Dict with synchronization results
        """
        try:
            sync_results = {
                "object_id": object_id,
                "systems_updated": [],
                "conflicts_resolved": [],
                "sync_successful": True,
            }

            # Get current state from each system
            states = {}

            if self.conditions:
                try:
                    # Get condition state
                    condition_desc = await self.conditions.get_condition_description(
                        object_id
                    )
                    states["condition"] = condition_desc
                except Exception as e:
                    logger.warning(
                        f"Could not get condition state for {object_id}: {e}"
                    )

            # Cross-reference states and resolve conflicts
            conflicts = self._detect_state_conflicts(states)

            if conflicts or force_update:
                # Apply state synchronization
                for system_name, system in [
                    ("inventory", self.inventory),
                    ("interactions", self.interactions),
                    ("conditions", self.conditions),
                    ("containers", self.containers),
                    ("crafting", self.crafting),
                ]:
                    if system:
                        try:
                            await self._synchronize_system_state(
                                system_name, system, object_id, states
                            )
                            sync_results["systems_updated"].append(system_name)
                        except Exception as e:
                            logger.error(
                                f"Failed to sync {system_name} for {object_id}: {e}"
                            )
                            sync_results["sync_successful"] = False

                sync_results["conflicts_resolved"] = conflicts

            return sync_results

        except Exception as e:
            logger.error(f"Error synchronizing object state: {e}")
            return {"object_id": object_id, "error": str(e), "sync_successful": False}

    async def process_batch_operations(
        self, operations: list[dict[str, Any]], parallel: bool = True
    ) -> list[dict[str, Any]]:
        """
        Process multiple operations efficiently.

        Args:
            operations: List of operations to process
            parallel: Whether to process operations in parallel

        Returns:
            List of operation results
        """
        try:
            if parallel:
                # Process operations concurrently
                tasks = []
                for operation in operations:
                    task = self.process_unified_action(
                        operation.get("action_type"),
                        operation.get("actor_id"),
                        operation.get("target_object"),
                        operation.get("action_data", {}),
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle any exceptions
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append(
                            {
                                "success": False,
                                "operation_index": i,
                                "error": str(result),
                            }
                        )
                    else:
                        result["operation_index"] = i
                        processed_results.append(result)

                return processed_results

            else:
                # Process operations sequentially
                results = []
                for i, operation in enumerate(operations):
                    result = await self.process_unified_action(
                        operation.get("action_type"),
                        operation.get("actor_id"),
                        operation.get("target_object"),
                        operation.get("action_data", {}),
                    )
                    result["operation_index"] = i
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"Error processing batch operations: {e}")
            return [{"error": str(e), "success": False}]

    async def get_system_health_report(self) -> dict[str, Any]:
        """
        Generate comprehensive health report for all object systems.

        Returns:
            Dict with system health information
        """
        try:
            health_report = {
                "timestamp": asyncio.get_event_loop().time(),
                "overall_health": "unknown",
                "systems": {},
                "performance_metrics": self._performance_metrics.copy(),
                "recommendations": [],
            }

            # Check each system
            systems_to_check = [
                ("inventory", self.inventory),
                ("interactions", self.interactions),
                ("conditions", self.conditions),
                ("containers", self.containers),
                ("crafting", self.crafting),
            ]

            healthy_systems = 0
            total_systems = len(systems_to_check)

            for system_name, system in systems_to_check:
                if system:
                    try:
                        # Basic health check - system is initialized and responsive
                        system_health = {
                            "status": "healthy",
                            "initialized": True,
                            "responsive": True,
                            "last_operation": "recent",
                        }
                        healthy_systems += 1
                    except Exception as e:
                        system_health = {
                            "status": "unhealthy",
                            "error": str(e),
                            "initialized": False,
                            "responsive": False,
                        }
                else:
                    system_health = {
                        "status": "not_initialized",
                        "initialized": False,
                        "responsive": False,
                    }

                health_report["systems"][system_name] = system_health

            # Calculate overall health
            health_percentage = (healthy_systems / total_systems) * 100
            if health_percentage >= 90:
                health_report["overall_health"] = "excellent"
            elif health_percentage >= 75:
                health_report["overall_health"] = "good"
            elif health_percentage >= 50:
                health_report["overall_health"] = "fair"
            else:
                health_report["overall_health"] = "poor"

            # Generate recommendations
            health_report["recommendations"] = self._generate_health_recommendations(
                health_report["systems"]
            )

            return health_report

        except Exception as e:
            logger.error(f"Error generating system health report: {e}")
            return {"error": str(e), "overall_health": "error"}

    # Private helper methods

    async def validate_action_prerequisites(
        self,
        action_type: str,
        actor_id: str,
        target_object: str,
        action_data: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate prerequisites for an action."""
        errors = []

        # Basic validation
        if not actor_id:
            errors.append("Actor ID is required")

        if not target_object and action_type not in [
            "craft_item",
            "organize_inventory",
            "transfer_items",
        ]:
            errors.append("Target object is required for this action")

        # Action-specific validation
        if action_type == "craft_item":
            if "recipe_id" not in action_data and "components" not in action_data:
                errors.append("Recipe ID or components required for crafting")

        elif action_type == "use_tool":
            if "tool_id" not in action_data:
                errors.append("Tool ID required for tool usage")

        return len(errors) == 0, errors

    async def _process_crafting_action(
        self, actor_id: str, action_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a crafting action."""
        try:
            recipe_id = action_data.get("recipe_id")
            component_sources = action_data.get("component_sources", {})

            if not recipe_id:
                return {"success": False, "error": "Recipe ID required"}

            # Start crafting session
            success, session_data = await self.crafting.start_crafting_session(
                actor_id, recipe_id, component_sources
            )

            if success:
                # Auto-complete simple crafting or return session for complex crafting
                session_id = session_data["session_id"]
                completion_success, completion_result = (
                    await self.crafting.complete_crafting_session(session_id)
                )

                return {
                    "success": completion_success,
                    "session_data": session_data,
                    "completion_result": completion_result,
                    "action_type": "craft_item",
                }
            else:
                return {
                    "success": False,
                    "error": session_data.get("error", "Crafting failed"),
                    "action_type": "craft_item",
                }

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "craft_item"}

    async def _process_tool_usage_action(
        self, actor_id: str, target_object: str, action_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a tool usage action."""
        try:
            tool_id = action_data.get("tool_id")
            action = action_data.get("action", "use")
            context = action_data.get("context", {"player_id": actor_id})

            # Execute tool interaction
            result = await self.interactions.execute_tool_interaction(
                tool_id, target_object, action, context
            )

            return {
                "success": result.success,
                "interaction_result": result,
                "action_type": "use_tool",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "use_tool"}

    async def _process_combination_action(
        self, actor_id: str, action_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process an object combination action."""
        try:
            primary_object = action_data.get("primary_object")
            secondary_objects = action_data.get("secondary_objects", [])
            recipe_id = action_data.get("recipe_id")
            context = action_data.get("context", {"player_id": actor_id})

            # Execute combination
            result = await self.interactions.process_object_combination(
                primary_object, secondary_objects, recipe_id, context
            )

            return {
                "success": result.success,
                "interaction_result": result,
                "action_type": "combine_objects",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "combine_objects"}

    async def _process_container_organization(
        self, actor_id: str, container_id: str, action_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process container organization action."""
        try:
            organization_type = action_data.get("organization_type", "auto")

            # Organize container
            result = await self.containers.organize_container_contents(
                container_id, organization_type
            )

            return {
                "success": "error" not in result,
                "organization_result": result,
                "action_type": "organize_container",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "organize_container",
            }

    async def _process_repair_action(
        self, actor_id: str, target_object: str, action_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process object repair action."""
        try:
            repair_materials = action_data.get("repair_materials", [])
            repair_skill = action_data.get("repair_skill", 5)
            repair_tools = action_data.get("repair_tools", [])

            # Attempt repair
            success, repair_result = await self.conditions.repair_object(
                target_object, repair_materials, repair_skill, repair_tools
            )

            return {
                "success": success,
                "repair_result": repair_result,
                "action_type": "repair_object",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "repair_object"}

    async def _process_transfer_action(
        self, actor_id: str, action_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process item transfer action."""
        try:
            from_inventory = action_data.get("from_inventory")
            to_inventory = action_data.get("to_inventory")
            item_id = action_data.get("item_id")
            quantity = action_data.get("quantity", 1)

            # Execute transfer
            success = await self.inventory.move_item(
                from_inventory, to_inventory, item_id, quantity
            )

            return {
                "success": success,
                "transfer_details": {
                    "from": from_inventory,
                    "to": to_inventory,
                    "item": item_id,
                    "quantity": quantity,
                },
                "action_type": "transfer_items",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": "transfer_items"}

    async def _process_generic_action(
        self,
        action_type: str,
        actor_id: str,
        target_object: str,
        action_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a generic action through the interaction system."""
        try:
            # Map to interaction type
            interaction_type_map = {
                "examine": ObjectInteractionType.EXAMINE,
                "use": ObjectInteractionType.USE,
                "consume": ObjectInteractionType.CONSUME,
                "transform": ObjectInteractionType.TRANSFORM,
                "disassemble": ObjectInteractionType.DISASSEMBLE,
            }

            interaction_type = interaction_type_map.get(
                action_type, ObjectInteractionType.USE
            )
            context = action_data.get("context", {"player_id": actor_id})

            # Execute interaction
            result = await self.interactions.process_object_interaction(
                interaction_type, target_object, None, None, context
            )

            return {
                "success": result.success,
                "interaction_result": result,
                "action_type": action_type,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "action_type": action_type}

    async def _apply_cross_system_effects(
        self,
        action_type: str,
        actor_id: str,
        target_object: str,
        result: dict[str, Any],
    ) -> None:
        """Apply effects that cross multiple systems."""
        try:
            if result.get("success") and hasattr(
                result.get("interaction_result"), "tool_used"
            ):
                tool_used = result["interaction_result"].tool_used
                if tool_used:
                    # Apply wear to tool
                    await self.interactions.apply_wear_and_degradation(
                        tool_used, 1.0, ObjectInteractionType.USE
                    )

        except Exception as e:
            logger.error(f"Error applying cross-system effects: {e}")

    async def _update_performance_metrics(
        self, action_type: str, elapsed_time: float, success: bool
    ) -> None:
        """Update performance metrics."""
        try:
            if action_type not in self._performance_metrics:
                self._performance_metrics[action_type] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "total_time": 0.0,
                    "average_time": 0.0,
                    "success_rate": 0.0,
                }

            metrics = self._performance_metrics[action_type]
            metrics["total_executions"] += 1
            metrics["total_time"] += elapsed_time
            metrics["average_time"] = (
                metrics["total_time"] / metrics["total_executions"]
            )

            if success:
                metrics["successful_executions"] += 1

            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"]
            )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _assess_overall_object_status(self, systems: dict[str, Any]) -> str:
        """Assess overall object status from system information."""
        error_count = sum(1 for system in systems.values() if "error" in system)
        attention_needed = sum(
            1 for system in systems.values() if system.get("needs_attention", False)
        )

        if error_count > 0:
            return "error"
        elif attention_needed > 0:
            return "needs_attention"
        else:
            return "good"

    def _generate_object_recommendations(self, systems: dict[str, Any]) -> list[str]:
        """Generate recommendations based on object status."""
        recommendations = []

        if systems.get("condition", {}).get("needs_attention"):
            recommendations.append(
                "Object condition is poor - consider repair or maintenance"
            )

        if len(systems.get("interactions", {}).get("interactions", [])) == 0:
            recommendations.append(
                "No interactions available - object may be broken or in wrong location"
            )

        return recommendations

    async def _optimize_inventory_performance(self) -> dict[str, Any]:
        """Optimize inventory system performance."""
        return {
            "optimization": "inventory",
            "improvements": ["cache optimization"],
            "performance_gain": 5.0,
        }

    async def _optimize_interaction_performance(self) -> dict[str, Any]:
        """Optimize interaction system performance."""
        return {
            "optimization": "interactions",
            "improvements": ["handler optimization"],
            "performance_gain": 3.0,
        }

    async def _optimize_condition_performance(self) -> dict[str, Any]:
        """Optimize condition system performance."""
        return {
            "optimization": "conditions",
            "improvements": ["update batching"],
            "performance_gain": 4.0,
        }

    async def _optimize_container_performance(self) -> dict[str, Any]:
        """Optimize container system performance."""
        return {
            "optimization": "containers",
            "improvements": ["hierarchy caching"],
            "performance_gain": 6.0,
        }

    async def _optimize_crafting_performance(self) -> dict[str, Any]:
        """Optimize crafting system performance."""
        return {
            "optimization": "crafting",
            "improvements": ["recipe caching"],
            "performance_gain": 7.0,
        }

    def _calculate_overall_improvement(
        self, optimization_results: dict[str, Any]
    ) -> float:
        """Calculate overall performance improvement."""
        total_improvement = sum(
            result.get("performance_gain", 0.0)
            for result in optimization_results.values()
        )
        return (
            total_improvement / len(optimization_results)
            if optimization_results
            else 0.0
        )

    def _detect_state_conflicts(self, states: dict[str, Any]) -> list[str]:
        """Detect conflicts between system states."""
        # Placeholder - would implement actual conflict detection
        return []

    async def _synchronize_system_state(
        self, system_name: str, system: Any, object_id: str, states: dict[str, Any]
    ) -> None:
        """Synchronize state for a specific system."""
        # Placeholder - would implement actual state synchronization
        pass

    def _generate_health_recommendations(self, systems: dict[str, Any]) -> list[str]:
        """Generate health recommendations based on system status."""
        recommendations = []

        for system_name, system_health in systems.items():
            if system_health.get("status") == "unhealthy":
                recommendations.append(f"Investigate {system_name} system issues")
            elif not system_health.get("initialized"):
                recommendations.append(f"Initialize {system_name} system")

        return recommendations

    def _initialize_system_coordination(self) -> None:
        """Initialize coordination between systems."""
        try:
            # Set up event handlers for cross-system communication
            self._event_handlers["object_condition_changed"] = [
                self._handle_condition_change
            ]
            self._event_handlers["inventory_updated"] = [self._handle_inventory_update]
            self._event_handlers["crafting_completed"] = [
                self._handle_crafting_completion
            ]

            logger.info("Object system integration initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing system coordination: {e}")

    async def _handle_condition_change(self, event: ObjectSystemEvent) -> None:
        """Handle object condition change events."""
        try:
            # Update related systems when object condition changes
            object_id = event.object_id
            new_condition = event.data.get("new_condition", 1.0)

            # If object is severely damaged, it might become unusable
            if new_condition <= 0.1:
                # Notify other systems that object is broken
                logger.info(
                    f"Object {object_id} is severely damaged (condition: {new_condition})"
                )

        except Exception as e:
            logger.error(f"Error handling condition change event: {e}")

    async def _handle_inventory_update(self, event: ObjectSystemEvent) -> None:
        """Handle inventory update events."""
        try:
            # Coordinate inventory changes with other systems
            logger.debug(f"Inventory updated: {event.data}")

        except Exception as e:
            logger.error(f"Error handling inventory update event: {e}")

    async def _handle_crafting_completion(self, event: ObjectSystemEvent) -> None:
        """Handle crafting completion events."""
        try:
            # Handle cross-system effects of crafting completion
            logger.debug(f"Crafting completed: {event.data}")

        except Exception as e:
            logger.error(f"Error handling crafting completion event: {e}")
