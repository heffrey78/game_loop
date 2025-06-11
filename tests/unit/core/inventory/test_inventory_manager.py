"""
Unit tests for InventoryManager.

Tests comprehensive inventory management including constraints, organization,
and multi-slot functionality.
"""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from src.game_loop.core.inventory.inventory_manager import (
    InventoryConstraint,
    InventoryConstraintType,
    InventoryManager,
)


class TestInventoryManager:
    """Test cases for InventoryManager functionality."""

    @pytest.fixture
    def inventory_manager(self):
        """Create inventory manager for testing."""
        object_manager = Mock()

        # Create a mock that returns item-specific properties
        async def mock_get_object_properties(item_id):
            properties = {
                "sword": {
                    "name": "Iron Sword",
                    "weight": 2.0,
                    "volume": 1.5,
                    "category": "weapon",
                    "description": "A sharp iron sword",
                },
                "shield": {
                    "name": "Wooden Shield",
                    "weight": 3.0,
                    "volume": 2.0,
                    "category": "armor",
                    "description": "A sturdy wooden shield",
                },
                "potion": {
                    "name": "Health Potion",
                    "weight": 0.5,
                    "volume": 0.3,
                    "category": "consumable",
                    "description": "A red healing potion",
                },
            }
            return properties.get(
                item_id,
                {
                    "name": item_id.replace("_", " ").title(),
                    "weight": 1.0,
                    "volume": 1.0,
                    "category": "misc",
                    "description": f"A {item_id}",
                },
            )

        object_manager.get_object_properties = mock_get_object_properties

        physics_engine = Mock()
        search_service = Mock()

        manager = InventoryManager(object_manager, physics_engine, search_service)
        return manager

    @pytest_asyncio.fixture
    async def test_inventory(self, inventory_manager):
        """Create a test inventory."""
        inventory_id = await inventory_manager.create_inventory("test_player")
        return inventory_id

    @pytest.mark.asyncio
    async def test_create_inventory(self, inventory_manager):
        """Test inventory creation."""
        inventory_id = await inventory_manager.create_inventory(
            "test_player", "default"
        )

        assert inventory_id.startswith("inv_test_player_")
        assert inventory_id in inventory_manager._inventories

        inventory = inventory_manager._inventories[inventory_id]
        assert inventory["owner_id"] == "test_player"
        assert inventory["template"] == "default"
        assert len(inventory["slots"]) == 20  # Default template has 20 slots

    @pytest.mark.asyncio
    async def test_add_item_success(self, inventory_manager, test_inventory):
        """Test successful item addition."""
        success, result = await inventory_manager.add_item(
            test_inventory, "test_item", 1
        )

        assert success is True
        assert result["item_id"] == "test_item"
        assert result["quantity"] == 1
        assert "slot" in result

    @pytest.mark.asyncio
    async def test_add_item_stacking(self, inventory_manager, test_inventory):
        """Test item stacking in same slot."""
        # Add first item
        await inventory_manager.add_item(test_inventory, "test_item", 1)

        # Add same item again - should stack
        success, result = await inventory_manager.add_item(
            test_inventory, "test_item", 2
        )

        assert success is True
        assert result["quantity"] == 2

        # Check that slot has correct total quantity
        inventory = inventory_manager._inventories[test_inventory]
        slot_id = result["slot"]
        assert inventory["slots"][slot_id].quantity == 3

    @pytest.mark.asyncio
    async def test_remove_item_success(self, inventory_manager, test_inventory):
        """Test successful item removal."""
        # Add item first
        await inventory_manager.add_item(test_inventory, "test_item", 5)

        # Remove some items
        success, result = await inventory_manager.remove_item(
            test_inventory, "test_item", 3
        )

        assert success is True
        assert result["item_id"] == "test_item"
        assert result["quantity"] == 3

        # Check remaining quantity
        inventory = inventory_manager._inventories[test_inventory]
        slot_with_item = None
        for slot in inventory["slots"].values():
            if slot.item_id == "test_item":
                slot_with_item = slot
                break

        assert slot_with_item is not None
        assert slot_with_item.quantity == 2

    @pytest.mark.asyncio
    async def test_remove_item_not_found(self, inventory_manager, test_inventory):
        """Test removing item that doesn't exist."""
        success, result = await inventory_manager.remove_item(
            test_inventory, "nonexistent_item", 1
        )

        assert success is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_remove_item_insufficient_quantity(
        self, inventory_manager, test_inventory
    ):
        """Test removing more items than available."""
        await inventory_manager.add_item(test_inventory, "test_item", 2)

        success, result = await inventory_manager.remove_item(
            test_inventory, "test_item", 5
        )

        assert success is False
        assert "Only 2 available" in result["error"]

    @pytest.mark.asyncio
    async def test_move_item_between_inventories(self, inventory_manager):
        """Test moving items between different inventories."""
        # Create two inventories
        inv1 = await inventory_manager.create_inventory("player1")
        inv2 = await inventory_manager.create_inventory("player2")

        # Add item to first inventory
        await inventory_manager.add_item(inv1, "test_item", 3)

        # Move item to second inventory
        success = await inventory_manager.move_item(inv1, inv2, "test_item", 2)

        assert success is True

        # Check quantities in both inventories
        inv1_data = inventory_manager._inventories[inv1]
        inv2_data = inventory_manager._inventories[inv2]

        # First inventory should have 1 remaining
        inv1_slot = None
        for slot in inv1_data["slots"].values():
            if slot.item_id == "test_item":
                inv1_slot = slot
                break
        assert inv1_slot.quantity == 1

        # Second inventory should have 2
        inv2_slot = None
        for slot in inv2_data["slots"].values():
            if slot.item_id == "test_item":
                inv2_slot = slot
                break
        assert inv2_slot.quantity == 2

    @pytest.mark.asyncio
    async def test_weight_constraint_validation(self, inventory_manager):
        """Test weight constraint enforcement."""
        # Create inventory with low weight limit
        custom_constraints = [
            InventoryConstraint(
                constraint_type=InventoryConstraintType.WEIGHT,
                limit=2.0,
                current=0.0,
                unit="kg",
                description="Low weight limit for testing",
            )
        ]

        inventory_id = await inventory_manager.create_inventory(
            "test_player", "default", custom_constraints
        )

        # Try to add item that would exceed weight limit
        # Mock heavy item
        inventory_manager.objects.get_object_properties = AsyncMock(
            return_value={
                "name": "Heavy Item",
                "weight": 5.0,
                "volume": 1.0,
                "category": "misc",
            }
        )

        success, result = await inventory_manager.add_item(
            inventory_id, "heavy_item", 1
        )

        assert success is False
        assert "Weight limit exceeded" in result["details"][0]

    @pytest.mark.asyncio
    async def test_search_inventory(self, inventory_manager, test_inventory):
        """Test inventory search functionality."""
        # Add some items
        await inventory_manager.add_item(test_inventory, "sword", 1)
        await inventory_manager.add_item(test_inventory, "shield", 1)
        await inventory_manager.add_item(test_inventory, "potion", 3)

        # Search for items
        results = await inventory_manager.search_inventory(test_inventory, "sword")

        assert len(results) == 1
        assert results[0]["item_id"] == "sword"
        assert results[0]["quantity"] == 1

    @pytest.mark.asyncio
    async def test_inventory_summary(self, inventory_manager, test_inventory):
        """Test inventory summary generation."""
        # Add some items
        await inventory_manager.add_item(test_inventory, "item1", 2)
        await inventory_manager.add_item(test_inventory, "item2", 3)

        summary = await inventory_manager.get_inventory_summary(test_inventory)

        assert summary["inventory_id"] == test_inventory
        assert summary["total_items"] == 5
        assert summary["slots_used"] == 2
        assert summary["slots_total"] == 20
        assert "constraints" in summary

    @pytest.mark.asyncio
    async def test_organize_inventory(self, inventory_manager, test_inventory):
        """Test inventory organization."""
        # Add some items to organize
        await inventory_manager.add_item(test_inventory, "item1", 1)
        await inventory_manager.add_item(test_inventory, "item2", 1)

        result = await inventory_manager.organize_inventory(test_inventory, "auto")

        assert "error" not in result
        assert result["strategy"] == "auto"

    @pytest.mark.asyncio
    async def test_constraint_validation(self, inventory_manager, test_inventory):
        """Test constraint validation system."""
        proposed_changes = {
            "add_item": {
                "item_id": "test_item",
                "quantity": 1,
                "weight": 1.0,
                "volume": 1.0,
            }
        }

        is_valid, violations = await inventory_manager.validate_constraints(
            test_inventory, proposed_changes
        )

        assert is_valid is True
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_inventory_effects_decay(self, inventory_manager, test_inventory):
        """Test inventory effects like item decay."""
        # Add perishable item
        inventory_manager.objects.get_object_properties = AsyncMock(
            return_value={
                "name": "Perishable Item",
                "weight": 1.0,
                "volume": 1.0,
                "category": "food",
                "perishable": True,
            }
        )

        await inventory_manager.add_item(test_inventory, "perishable_item", 1)

        # Apply decay effect
        result = await inventory_manager.apply_inventory_effects(
            test_inventory, "decay", {"decay_rate": 0.1, "time_elapsed": 1.0}
        )

        assert result["effect_type"] == "decay"
        assert len(result["affected_items"]) == 1
        assert result["affected_items"][0]["item_id"] == "perishable_item"

    @pytest.mark.asyncio
    async def test_calculate_carry_capacity(self, inventory_manager):
        """Test carry capacity calculation."""
        capacity = await inventory_manager.calculate_carry_capacity("test_player")

        assert "weight_capacity" in capacity
        assert "volume_capacity" in capacity
        assert "count_capacity" in capacity
        assert capacity["weight_capacity"] > 0
        assert capacity["volume_capacity"] > 0

    @pytest.mark.asyncio
    async def test_custom_constraint_validator(self, inventory_manager):
        """Test registering custom constraint validators."""

        # Register custom validator
        async def custom_validator(constraint, changes):
            return True, None

        inventory_manager.register_constraint_validator(
            InventoryConstraintType.SPECIAL, custom_validator
        )

        # Verify it's registered
        assert (
            InventoryConstraintType.SPECIAL in inventory_manager._constraint_validators
        )

    @pytest.mark.asyncio
    async def test_inventory_history_tracking(self, inventory_manager, test_inventory):
        """Test that inventory actions are tracked in history."""
        # Perform some actions
        await inventory_manager.add_item(test_inventory, "test_item", 1)
        await inventory_manager.remove_item(test_inventory, "test_item", 1)

        # Check history
        history = inventory_manager._inventory_history[test_inventory]
        assert len(history) == 2
        assert history[0]["action_type"] == "add_item"
        assert history[1]["action_type"] == "remove_item"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_inventory(self, inventory_manager):
        """Test error handling for invalid inventory operations."""
        success, result = await inventory_manager.add_item(
            "invalid_inventory", "item", 1
        )

        assert success is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_slot_locking_functionality(self, inventory_manager, test_inventory):
        """Test slot locking functionality."""
        inventory = inventory_manager._inventories[test_inventory]

        # Lock a slot
        first_slot_id = list(inventory["slots"].keys())[0]
        inventory["slots"][first_slot_id].locked = True

        # Try to add item - should skip locked slot
        success, result = await inventory_manager.add_item(
            test_inventory, "test_item", 1
        )

        assert success is True
        assert result["slot"] != first_slot_id
