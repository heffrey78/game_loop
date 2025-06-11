"""
Inventory Manager for comprehensive item handling with realistic constraints.

This module provides sophisticated inventory management with weight/volume constraints,
multi-container support, smart organization, and advanced search capabilities.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InventoryConstraintType(Enum):
    """Types of constraints that can be applied to inventories."""

    WEIGHT = "weight"
    VOLUME = "volume"
    COUNT = "count"
    CATEGORY = "category"
    SPECIAL = "special"


@dataclass
class InventoryConstraint:
    """Defines a constraint for inventory capacity or organization."""

    constraint_type: InventoryConstraintType
    limit: float
    current: float
    unit: str
    description: str


@dataclass
class InventorySlot:
    """Represents a single slot in an inventory."""

    slot_id: str
    item_id: str | None = None
    quantity: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    constraints: list[InventoryConstraintType] = field(default_factory=list)
    locked: bool = False


class InventoryManager:
    """
    Manage inventories with realistic constraints and advanced features.

    This class provides comprehensive inventory management including:
    - Weight, volume, and count constraints
    - Multi-slot organization with metadata
    - Smart item placement and organization
    - Advanced search and filtering
    - Constraint validation and enforcement
    """

    def __init__(
        self,
        object_manager: Any = None,
        physics_engine: Any = None,
        search_service: Any = None,
    ):
        """
        Initialize the inventory manager.

        Args:
            object_manager: Manager for object data and properties
            physics_engine: Physics engine for weight/volume calculations
            search_service: Search service for inventory queries
        """
        self.objects = object_manager
        self.physics = physics_engine
        self.search = search_service
        self._inventories: dict[str, dict[str, Any]] = {}
        self._inventory_templates: dict[str, dict[str, Any]] = {}
        self._constraint_validators: dict[InventoryConstraintType, Callable] = {}
        self._inventory_history: dict[str, list[dict[str, Any]]] = {}
        self._initialize_default_constraints()

    async def create_inventory(
        self,
        owner_id: str,
        template_name: str = "default",
        custom_constraints: list[InventoryConstraint] | None = None,
    ) -> str:
        """
        Create a new inventory for an entity.

        Args:
            owner_id: ID of the entity that owns this inventory
            template_name: Template to use for inventory creation
            custom_constraints: Additional constraints to apply

        Returns:
            str: Unique inventory ID
        """
        try:
            inventory_id = f"inv_{owner_id}_{uuid.uuid4().hex[:8]}"

            # Load template or create default
            template = self._inventory_templates.get(
                template_name, self._create_default_template()
            )

            # Create inventory structure
            inventory = {
                "id": inventory_id,
                "owner_id": owner_id,
                "template": template_name,
                "slots": {},
                "constraints": {},
                "metadata": {
                    "created_at": asyncio.get_event_loop().time(),
                    "total_items": 0,
                    "organization_strategy": "auto",
                },
                "quick_access": {},  # For frequently accessed items
            }

            # Initialize slots based on template
            for slot_config in template.get("slots", []):
                slot = InventorySlot(
                    slot_id=slot_config["id"],
                    constraints=slot_config.get("constraints", []),
                    metadata=slot_config.get("metadata", {}),
                )
                inventory["slots"][slot.slot_id] = slot

            # Set up constraints
            for constraint in template.get("constraints", []):
                inventory["constraints"][constraint["type"]] = InventoryConstraint(
                    constraint_type=InventoryConstraintType(constraint["type"]),
                    limit=constraint["limit"],
                    current=0.0,
                    unit=constraint["unit"],
                    description=constraint["description"],
                )

            # Add custom constraints
            if custom_constraints:
                for constraint in custom_constraints:
                    inventory["constraints"][
                        constraint.constraint_type.value
                    ] = constraint

            self._inventories[inventory_id] = inventory
            self._inventory_history[inventory_id] = []

            logger.info(
                f"Created inventory {inventory_id} for {owner_id} using template {template_name}"
            )
            return inventory_id

        except Exception as e:
            logger.error(f"Error creating inventory: {e}")
            raise

    async def add_item(
        self,
        inventory_id: str,
        item_id: str,
        quantity: int = 1,
        target_slot: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Add item to inventory with constraint validation.

        Args:
            inventory_id: Target inventory ID
            item_id: ID of item to add
            quantity: Number of items to add
            target_slot: Specific slot to add to (optional)

        Returns:
            Tuple of (success, result_info)
        """
        try:
            if inventory_id not in self._inventories:
                return False, {"error": f"Inventory {inventory_id} not found"}

            inventory = self._inventories[inventory_id]

            # Get item properties
            item_properties = await self._get_item_properties(item_id)
            if not item_properties:
                return False, {"error": f"Item {item_id} not found"}

            # Calculate total requirements for this addition
            total_weight = item_properties.get("weight", 0.0) * quantity
            total_volume = item_properties.get("volume", 0.0) * quantity

            # Validate constraints
            is_valid, constraint_errors = await self.validate_constraints(
                inventory_id,
                {
                    "add_item": {
                        "item_id": item_id,
                        "quantity": quantity,
                        "weight": total_weight,
                        "volume": total_volume,
                    }
                },
            )

            if not is_valid:
                return False, {
                    "error": "Constraint violations",
                    "details": constraint_errors,
                }

            # Find optimal slot
            if not target_slot:
                target_slot = await self._find_optimal_slot(inventory_id, item_id)

            if not target_slot:
                return False, {"error": "No available slot found"}

            # Add item to slot
            slot = inventory["slots"][target_slot]
            if slot.item_id == item_id:
                # Stack with existing item
                slot.quantity += quantity
            elif slot.item_id is None:
                # Place in empty slot
                slot.item_id = item_id
                slot.quantity = quantity
            else:
                return False, {"error": f"Slot {target_slot} already occupied"}

            # Update constraints
            await self._update_constraint_values(
                inventory_id, total_weight, total_volume, quantity
            )

            # Update metadata
            inventory["metadata"]["total_items"] += quantity

            # Record in history
            await self._record_inventory_action(
                inventory_id,
                "add_item",
                {
                    "item_id": item_id,
                    "quantity": quantity,
                    "slot": target_slot,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

            return True, {
                "item_id": item_id,
                "quantity": quantity,
                "slot": target_slot,
                "total_weight": total_weight,
                "total_volume": total_volume,
            }

        except Exception as e:
            logger.error(f"Error adding item to inventory: {e}")
            return False, {"error": str(e)}

    async def remove_item(
        self,
        inventory_id: str,
        item_id: str,
        quantity: int = 1,
        from_slot: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Remove item from inventory.

        Args:
            inventory_id: Source inventory ID
            item_id: ID of item to remove
            quantity: Number of items to remove
            from_slot: Specific slot to remove from (optional)

        Returns:
            Tuple of (success, result_info)
        """
        try:
            if inventory_id not in self._inventories:
                return False, {"error": f"Inventory {inventory_id} not found"}

            inventory = self._inventories[inventory_id]

            # Find item in inventory
            target_slots = []
            if from_slot:
                if (
                    from_slot in inventory["slots"]
                    and inventory["slots"][from_slot].item_id == item_id
                ):
                    target_slots = [from_slot]
            else:
                # Find all slots containing this item
                target_slots = [
                    slot_id
                    for slot_id, slot in inventory["slots"].items()
                    if slot.item_id == item_id and slot.quantity > 0
                ]

            if not target_slots:
                return False, {"error": f"Item {item_id} not found in inventory"}

            # Calculate what we can actually remove
            available_quantity = sum(
                inventory["slots"][slot_id].quantity for slot_id in target_slots
            )

            if available_quantity < quantity:
                return False, {
                    "error": f"Only {available_quantity} available, cannot remove {quantity}"
                }

            # Get item properties for constraint updates
            item_properties = await self._get_item_properties(item_id)
            unit_weight = item_properties.get("weight", 0.0)
            unit_volume = item_properties.get("volume", 0.0)

            # Remove items from slots
            remaining_to_remove = quantity
            removed_from_slots = []

            for slot_id in target_slots:
                if remaining_to_remove <= 0:
                    break

                slot = inventory["slots"][slot_id]
                remove_from_this_slot = min(slot.quantity, remaining_to_remove)

                slot.quantity -= remove_from_this_slot
                remaining_to_remove -= remove_from_this_slot

                removed_from_slots.append(
                    {"slot_id": slot_id, "quantity": remove_from_this_slot}
                )

                # Clear slot if empty
                if slot.quantity == 0:
                    slot.item_id = None
                    slot.metadata.clear()

            # Update constraints
            total_weight_removed = unit_weight * quantity
            total_volume_removed = unit_volume * quantity
            await self._update_constraint_values(
                inventory_id, -total_weight_removed, -total_volume_removed, -quantity
            )

            # Update metadata
            inventory["metadata"]["total_items"] -= quantity

            # Record in history
            await self._record_inventory_action(
                inventory_id,
                "remove_item",
                {
                    "item_id": item_id,
                    "quantity": quantity,
                    "slots": removed_from_slots,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

            return True, {
                "item_id": item_id,
                "quantity": quantity,
                "slots": removed_from_slots,
                "weight_removed": total_weight_removed,
                "volume_removed": total_volume_removed,
            }

        except Exception as e:
            logger.error(f"Error removing item from inventory: {e}")
            return False, {"error": str(e)}

    async def move_item(
        self,
        from_inventory: str,
        to_inventory: str,
        item_id: str,
        quantity: int = 1,
        to_slot: str | None = None,
    ) -> bool:
        """
        Move item between inventories.

        Args:
            from_inventory: Source inventory ID
            to_inventory: Target inventory ID
            item_id: ID of item to move
            quantity: Number of items to move
            to_slot: Specific target slot (optional)

        Returns:
            bool: Success of move operation
        """
        try:
            # Remove from source
            remove_success, remove_result = await self.remove_item(
                from_inventory, item_id, quantity
            )
            if not remove_success:
                return False

            # Add to target
            add_success, add_result = await self.add_item(
                to_inventory, item_id, quantity, to_slot
            )
            if not add_success:
                # Rollback - add back to source
                await self.add_item(from_inventory, item_id, quantity)
                return False

            # Record cross-inventory transfer
            await self._record_inventory_action(
                from_inventory,
                "transfer_out",
                {
                    "item_id": item_id,
                    "quantity": quantity,
                    "target_inventory": to_inventory,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

            await self._record_inventory_action(
                to_inventory,
                "transfer_in",
                {
                    "item_id": item_id,
                    "quantity": quantity,
                    "source_inventory": from_inventory,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

            return True

        except Exception as e:
            logger.error(f"Error moving item between inventories: {e}")
            return False

    async def organize_inventory(
        self, inventory_id: str, strategy: str = "auto"
    ) -> dict[str, Any]:
        """
        Automatically organize inventory contents.

        Args:
            inventory_id: Inventory to organize
            strategy: Organization strategy to use

        Returns:
            Dict with organization results
        """
        try:
            if inventory_id not in self._inventories:
                return {"error": f"Inventory {inventory_id} not found"}

            inventory = self._inventories[inventory_id]

            if strategy == "auto":
                return await self._auto_organize_inventory(inventory_id)
            elif strategy == "category":
                return await self._organize_by_category(inventory_id)
            elif strategy == "weight":
                return await self._organize_by_weight(inventory_id)
            elif strategy == "frequency":
                return await self._organize_by_usage_frequency(inventory_id)
            else:
                return {"error": f"Unknown organization strategy: {strategy}"}

        except Exception as e:
            logger.error(f"Error organizing inventory: {e}")
            return {"error": str(e)}

    async def search_inventory(
        self, inventory_id: str, query: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for items within an inventory.

        Args:
            inventory_id: Inventory to search
            query: Search query string
            filters: Additional filters to apply

        Returns:
            List of matching items with slot information
        """
        try:
            if inventory_id not in self._inventories:
                return []

            inventory = self._inventories[inventory_id]
            results = []

            for slot_id, slot in inventory["slots"].items():
                if not slot.item_id:
                    continue

                item_properties = await self._get_item_properties(slot.item_id)
                if not item_properties:
                    continue

                # Basic text matching
                item_name = item_properties.get("name", slot.item_id).lower()
                item_description = item_properties.get("description", "").lower()

                if query.lower() in item_name or query.lower() in item_description:
                    match = {
                        "slot_id": slot_id,
                        "item_id": slot.item_id,
                        "quantity": slot.quantity,
                        "item_name": item_properties.get("name", slot.item_id),
                        "item_properties": item_properties,
                        "metadata": slot.metadata,
                    }

                    # Apply filters
                    if self._passes_filters(match, filters):
                        results.append(match)

            return results

        except Exception as e:
            logger.error(f"Error searching inventory: {e}")
            return []

    async def get_inventory_summary(self, inventory_id: str) -> dict[str, Any]:
        """
        Get comprehensive summary of inventory state.

        Args:
            inventory_id: Inventory to summarize

        Returns:
            Dict containing inventory summary
        """
        try:
            if inventory_id not in self._inventories:
                return {"error": f"Inventory {inventory_id} not found"}

            inventory = self._inventories[inventory_id]

            # Count items and calculate totals
            total_items = 0
            total_weight = 0.0
            total_volume = 0.0
            item_categories = {}

            for slot in inventory["slots"].values():
                if slot.item_id and slot.quantity > 0:
                    total_items += slot.quantity

                    item_properties = await self._get_item_properties(slot.item_id)
                    if item_properties:
                        total_weight += (
                            item_properties.get("weight", 0.0) * slot.quantity
                        )
                        total_volume += (
                            item_properties.get("volume", 0.0) * slot.quantity
                        )

                        category = item_properties.get("category", "misc")
                        item_categories[category] = (
                            item_categories.get(category, 0) + slot.quantity
                        )

            # Constraint utilization
            constraints_status = {}
            for constraint_type, constraint in inventory["constraints"].items():
                utilization = (
                    constraint.current / constraint.limit
                    if constraint.limit > 0
                    else 0.0
                )
                constraints_status[constraint_type] = {
                    "current": constraint.current,
                    "limit": constraint.limit,
                    "utilization": utilization,
                    "unit": constraint.unit,
                }

            return {
                "inventory_id": inventory_id,
                "owner_id": inventory["owner_id"],
                "total_items": total_items,
                "total_weight": total_weight,
                "total_volume": total_volume,
                "item_categories": item_categories,
                "constraints": constraints_status,
                "slots_used": sum(
                    1 for slot in inventory["slots"].values() if slot.item_id
                ),
                "slots_total": len(inventory["slots"]),
                "metadata": inventory["metadata"],
            }

        except Exception as e:
            logger.error(f"Error getting inventory summary: {e}")
            return {"error": str(e)}

    async def validate_constraints(
        self, inventory_id: str, proposed_changes: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate proposed inventory changes against constraints.

        Args:
            inventory_id: Inventory to validate
            proposed_changes: Changes to validate

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        try:
            if inventory_id not in self._inventories:
                return False, [f"Inventory {inventory_id} not found"]

            inventory = self._inventories[inventory_id]
            violations = []

            # Check each constraint type
            for constraint_type, constraint in inventory["constraints"].items():
                validator = self._constraint_validators.get(
                    InventoryConstraintType(constraint_type)
                )

                if validator:
                    is_valid, error_msg = await validator(constraint, proposed_changes)
                    if not is_valid and error_msg:
                        violations.append(error_msg)

            return len(violations) == 0, violations

        except Exception as e:
            logger.error(f"Error validating constraints: {e}")
            return False, [str(e)]

    async def calculate_carry_capacity(self, owner_id: str) -> dict[str, float]:
        """
        Calculate total carrying capacity for an entity.

        Args:
            owner_id: Entity to calculate capacity for

        Returns:
            Dict with capacity information
        """
        try:
            # Base capacity (would be determined by entity stats)
            base_weight_capacity = 50.0  # kg
            base_volume_capacity = 100.0  # liters

            # TODO: Factor in entity strength, equipment, skills, etc.
            # This would integrate with the player state and skill systems

            return {
                "weight_capacity": base_weight_capacity,
                "volume_capacity": base_volume_capacity,
                "count_capacity": 50,  # Number of distinct items
            }

        except Exception as e:
            logger.error(f"Error calculating carry capacity: {e}")
            return {}

    async def apply_inventory_effects(
        self, inventory_id: str, effect_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply ongoing effects to inventory (decay, temperature, etc.).

        Args:
            inventory_id: Inventory to affect
            effect_type: Type of effect to apply
            parameters: Effect parameters

        Returns:
            Dict with effect results
        """
        try:
            if inventory_id not in self._inventories:
                return {"error": f"Inventory {inventory_id} not found"}

            inventory = self._inventories[inventory_id]
            affected_items = []

            if effect_type == "decay":
                # Apply decay to perishable items
                decay_rate = parameters.get("decay_rate", 0.01)
                time_elapsed = parameters.get("time_elapsed", 1.0)

                for slot in inventory["slots"].values():
                    if slot.item_id:
                        item_properties = await self._get_item_properties(slot.item_id)
                        if item_properties and item_properties.get("perishable", False):
                            current_condition = slot.metadata.get("condition", 1.0)
                            new_condition = max(
                                0.0, current_condition - (decay_rate * time_elapsed)
                            )
                            slot.metadata["condition"] = new_condition

                            affected_items.append(
                                {
                                    "item_id": slot.item_id,
                                    "slot_id": slot.slot_id,
                                    "old_condition": current_condition,
                                    "new_condition": new_condition,
                                }
                            )

            return {
                "effect_type": effect_type,
                "affected_items": affected_items,
                "parameters": parameters,
            }

        except Exception as e:
            logger.error(f"Error applying inventory effects: {e}")
            return {"error": str(e)}

    def register_constraint_validator(
        self, constraint_type: InventoryConstraintType, validator: Callable
    ) -> None:
        """
        Register custom constraint validation logic.

        Args:
            constraint_type: Type of constraint to validate
            validator: Validation function
        """
        self._constraint_validators[constraint_type] = validator

    # Private helper methods

    async def _find_optimal_slot(self, inventory_id: str, item_id: str) -> str | None:
        """Find the optimal slot for an item in inventory."""
        try:
            inventory = self._inventories[inventory_id]
            item_properties = await self._get_item_properties(item_id)

            # First try to stack with existing items
            for slot_id, slot in inventory["slots"].items():
                if slot.item_id == item_id and not slot.locked:
                    return slot_id

            # Find empty slot with compatible constraints
            for slot_id, slot in inventory["slots"].items():
                if slot.item_id is None and not slot.locked:
                    # Check if item type is compatible with slot constraints
                    item_category = item_properties.get("category", "misc")
                    if not slot.constraints or self._check_slot_compatibility(
                        slot, item_category
                    ):
                        return slot_id

            return None

        except Exception as e:
            logger.error(f"Error finding optimal slot: {e}")
            return None

    async def _update_constraint_values(
        self,
        inventory_id: str,
        weight_delta: float,
        volume_delta: float,
        count_delta: int,
    ) -> None:
        """Update constraint current values."""
        try:
            inventory = self._inventories[inventory_id]

            if "weight" in inventory["constraints"]:
                inventory["constraints"]["weight"].current += weight_delta

            if "volume" in inventory["constraints"]:
                inventory["constraints"]["volume"].current += volume_delta

            if "count" in inventory["constraints"]:
                inventory["constraints"]["count"].current += count_delta

        except Exception as e:
            logger.error(f"Error updating constraint values: {e}")

    async def _record_inventory_action(
        self, inventory_id: str, action_type: str, action_data: dict[str, Any]
    ) -> None:
        """Record an action in inventory history."""
        try:
            if inventory_id not in self._inventory_history:
                self._inventory_history[inventory_id] = []

            self._inventory_history[inventory_id].append(
                {
                    "action_type": action_type,
                    "data": action_data,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

            # Limit history size
            max_history = 1000
            if len(self._inventory_history[inventory_id]) > max_history:
                self._inventory_history[inventory_id] = self._inventory_history[
                    inventory_id
                ][-max_history:]

        except Exception as e:
            logger.error(f"Error recording inventory action: {e}")

    async def _get_item_properties(self, item_id: str) -> dict[str, Any] | None:
        """Get properties for an item."""
        try:
            if self.objects:
                return await self.objects.get_object_properties(item_id)
            else:
                # Fallback for testing - return basic properties
                return {
                    "name": item_id.replace("_", " ").title(),
                    "weight": 1.0,
                    "volume": 1.0,
                    "category": "misc",
                    "description": f"A {item_id}",
                }

        except Exception as e:
            logger.error(f"Error getting item properties: {e}")
            return None

    def _check_slot_compatibility(
        self, slot: InventorySlot, item_category: str
    ) -> bool:
        """Check if an item category is compatible with slot constraints."""
        if InventoryConstraintType.CATEGORY in slot.constraints:
            allowed_categories = slot.metadata.get("allowed_categories", [])
            return item_category in allowed_categories
        return True

    def _passes_filters(
        self, item_match: dict[str, Any], filters: dict[str, Any] | None
    ) -> bool:
        """Check if an item match passes the given filters."""
        if not filters:
            return True

        # Category filter
        if "category" in filters:
            item_category = item_match["item_properties"].get("category", "misc")
            if item_category not in filters["category"]:
                return False

        # Weight range filter
        if "weight_range" in filters:
            weight = item_match["item_properties"].get("weight", 0.0)
            min_weight, max_weight = filters["weight_range"]
            if not (min_weight <= weight <= max_weight):
                return False

        # Quantity filter
        if "min_quantity" in filters:
            if item_match["quantity"] < filters["min_quantity"]:
                return False

        return True

    async def _auto_organize_inventory(self, inventory_id: str) -> dict[str, Any]:
        """Automatically organize inventory using smart strategy."""
        # This would implement intelligent organization based on:
        # - Item categories
        # - Usage frequency
        # - Item weight/size
        # - Player preferences
        return {"strategy": "auto", "changes_made": 0}

    async def _organize_by_category(self, inventory_id: str) -> dict[str, Any]:
        """Organize inventory by item categories."""
        return {"strategy": "category", "changes_made": 0}

    async def _organize_by_weight(self, inventory_id: str) -> dict[str, Any]:
        """Organize inventory by item weight."""
        return {"strategy": "weight", "changes_made": 0}

    async def _organize_by_usage_frequency(self, inventory_id: str) -> dict[str, Any]:
        """Organize inventory by item usage frequency."""
        return {"strategy": "frequency", "changes_made": 0}

    def _create_default_template(self) -> dict[str, Any]:
        """Create default inventory template."""
        return {
            "name": "default",
            "slots": [
                {"id": f"slot_{i}", "constraints": [], "metadata": {}}
                for i in range(20)  # 20 default slots
            ],
            "constraints": [
                {
                    "type": "weight",
                    "limit": 50.0,
                    "unit": "kg",
                    "description": "Maximum weight capacity",
                },
                {
                    "type": "volume",
                    "limit": 100.0,
                    "unit": "liters",
                    "description": "Maximum volume capacity",
                },
                {
                    "type": "count",
                    "limit": 20,
                    "unit": "items",
                    "description": "Maximum number of distinct items",
                },
            ],
        }

    def _initialize_default_constraints(self) -> None:
        """Initialize default constraint validators."""

        async def validate_weight_constraint(
            constraint: InventoryConstraint, changes: dict[str, Any]
        ) -> tuple[bool, str | None]:
            """Validate weight constraint."""
            if "add_item" in changes:
                additional_weight = changes["add_item"].get("weight", 0.0)
                new_total = constraint.current + additional_weight
                if new_total > constraint.limit:
                    return (
                        False,
                        f"Weight limit exceeded: {new_total:.1f}/{constraint.limit:.1f} {constraint.unit}",
                    )
            return True, None

        async def validate_volume_constraint(
            constraint: InventoryConstraint, changes: dict[str, Any]
        ) -> tuple[bool, str | None]:
            """Validate volume constraint."""
            if "add_item" in changes:
                additional_volume = changes["add_item"].get("volume", 0.0)
                new_total = constraint.current + additional_volume
                if new_total > constraint.limit:
                    return (
                        False,
                        f"Volume limit exceeded: {new_total:.1f}/{constraint.limit:.1f} {constraint.unit}",
                    )
            return True, None

        async def validate_count_constraint(
            constraint: InventoryConstraint, changes: dict[str, Any]
        ) -> tuple[bool, str | None]:
            """Validate count constraint."""
            if "add_item" in changes:
                additional_count = 1  # Each distinct item counts as 1
                new_total = constraint.current + additional_count
                if new_total > constraint.limit:
                    return (
                        False,
                        f"Item count limit exceeded: {new_total}/{constraint.limit} {constraint.unit}",
                    )
            return True, None

        self._constraint_validators[InventoryConstraintType.WEIGHT] = (
            validate_weight_constraint
        )
        self._constraint_validators[InventoryConstraintType.VOLUME] = (
            validate_volume_constraint
        )
        self._constraint_validators[InventoryConstraintType.COUNT] = (
            validate_count_constraint
        )
