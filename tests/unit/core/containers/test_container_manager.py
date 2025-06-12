"""
Unit tests for ContainerManager.

Tests container creation, access control, nested hierarchies, and organization.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from game_loop.core.containers.container_manager import (
    ContainerManager,
    ContainerSpecification,
    ContainerType,
)


class TestContainerManager:
    """Test cases for ContainerManager functionality."""

    @pytest.fixture
    def container_manager(self):
        """Create container manager for testing."""
        inventory_manager = Mock()
        inventory_manager.create_inventory = AsyncMock(return_value="test_inventory_id")
        inventory_manager.add_item = AsyncMock(
            return_value=(True, {"item_id": "test_item"})
        )
        inventory_manager.remove_item = AsyncMock(
            return_value=(True, {"item_id": "test_item"})
        )
        inventory_manager.get_inventory_summary = AsyncMock(return_value={"items": []})
        inventory_manager.search_inventory = AsyncMock(return_value=[])
        inventory_manager.move_item = AsyncMock(return_value=True)
        inventory_manager.apply_inventory_effects = AsyncMock(
            return_value={"effect": "preservation", "items_affected": 0}
        )

        object_manager = Mock()
        physics_engine = Mock()

        manager = ContainerManager(inventory_manager, object_manager, physics_engine)
        return manager

    @pytest.fixture
    def basic_container_spec(self):
        """Create basic container specification for testing."""
        return ContainerSpecification(
            container_type=ContainerType.GENERAL,
            capacity_slots=10,
            weight_limit=50.0,
            volume_limit=100.0,
            access_restrictions=[],
            organization_rules={},
            special_properties={},
        )

    @pytest.mark.asyncio
    async def test_create_container_success(
        self, container_manager, basic_container_spec
    ):
        """Test successful container creation."""
        success = await container_manager.create_container(
            "test_container", basic_container_spec, "test_owner"
        )

        assert success is True
        assert "test_container" in container_manager._container_registry

        container_data = container_manager._container_registry["test_container"]
        assert container_data["owner_id"] == "test_owner"
        assert container_data["type"] == ContainerType.GENERAL
        assert container_data["inventory_id"] == "test_inventory_id"

    @pytest.mark.asyncio
    async def test_create_container_duplicate_id(
        self, container_manager, basic_container_spec
    ):
        """Test creating container with duplicate ID."""
        # Create first container
        await container_manager.create_container("test_container", basic_container_spec)

        # Try to create another with same ID
        success = await container_manager.create_container(
            "test_container", basic_container_spec
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_open_container_success(
        self, container_manager, basic_container_spec
    ):
        """Test successful container opening."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        success, result = await container_manager.open_container(
            "test_container",
            "owner",  # Owner should have access
            {"location": "same_room"},
        )

        assert success is True
        assert "contents" in result
        assert "container_info" in result

    @pytest.mark.asyncio
    async def test_open_container_access_denied(
        self, container_manager, basic_container_spec
    ):
        """Test container opening with access denied."""
        # Create private container
        private_spec = ContainerSpecification(
            container_type=ContainerType.SAFE,
            capacity_slots=5,
            weight_limit=25.0,
            volume_limit=50.0,
            access_restrictions=["private"],
            organization_rules={},
            special_properties={"security_level": 5},
        )

        await container_manager.create_container(
            "private_container", private_spec, "owner"
        )

        success, result = await container_manager.open_container(
            "private_container", "other_user", {}  # Not the owner
        )

        assert success is False
        assert "access denied" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_open_container_already_open(
        self, container_manager, basic_container_spec
    ):
        """Test opening container that's already open."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        # Open container first time
        await container_manager.open_container("test_container", "owner", {})

        # Try to open again
        success, result = await container_manager.open_container(
            "test_container", "owner", {}
        )

        assert success is True
        assert "already open" in result["message"]

    @pytest.mark.asyncio
    async def test_close_container(self, container_manager, basic_container_spec):
        """Test container closing."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        # Open container first
        await container_manager.open_container("test_container", "owner", {})

        # Close container
        success = await container_manager.close_container("test_container", "owner")

        assert success is True

        # Check that container is closed
        container_state = container_manager._container_states["test_container"]
        assert container_state["open"] is False

    @pytest.mark.asyncio
    async def test_place_item_in_container(
        self, container_manager, basic_container_spec
    ):
        """Test placing item in container."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )
        await container_manager.open_container("test_container", "owner", {})

        success, result = await container_manager.place_item_in_container(
            "test_container", "test_item", 2, "auto"
        )

        assert success is True
        # Should have called inventory manager's add_item
        container_manager.inventory.add_item.assert_called()

    @pytest.mark.asyncio
    async def test_place_item_container_closed(
        self, container_manager, basic_container_spec
    ):
        """Test placing item in closed container."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )
        # Don't open the container

        success, result = await container_manager.place_item_in_container(
            "test_container", "test_item", 1
        )

        assert success is False
        assert "not open" in result["error"]

    @pytest.mark.asyncio
    async def test_retrieve_item_from_container(
        self, container_manager, basic_container_spec
    ):
        """Test retrieving item from container."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )
        await container_manager.open_container("test_container", "owner", {})

        success, result = await container_manager.retrieve_item_from_container(
            "test_container", "test_item", 1, "owner"
        )

        assert success is True
        # Should have called inventory manager's remove_item
        container_manager.inventory.remove_item.assert_called()

    @pytest.mark.asyncio
    async def test_organize_container_contents(
        self, container_manager, basic_container_spec
    ):
        """Test container content organization."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        result = await container_manager.organize_container_contents(
            "test_container", "auto"
        )

        assert "error" not in result
        assert result["strategy"] == "auto"

    @pytest.mark.asyncio
    async def test_get_container_hierarchy(
        self, container_manager, basic_container_spec
    ):
        """Test getting container hierarchy."""
        await container_manager.create_container(
            "root_container", basic_container_spec, "owner"
        )

        hierarchy = await container_manager.get_container_hierarchy("root_container", 3)

        assert "hierarchy" in hierarchy
        assert hierarchy["root_container"] == "root_container"
        assert "total_containers" in hierarchy
        assert "total_items" in hierarchy

    @pytest.mark.asyncio
    async def test_search_container_contents(
        self, container_manager, basic_container_spec
    ):
        """Test searching container contents."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        results = await container_manager.search_container_contents(
            "test_container", "sword", recursive=False
        )

        # Should return list (might be empty)
        assert isinstance(results, list)
        # Should have called inventory search
        container_manager.inventory.search_inventory.assert_called()

    @pytest.mark.asyncio
    async def test_search_container_contents_recursive(
        self, container_manager, basic_container_spec
    ):
        """Test recursive container content search."""
        await container_manager.create_container(
            "parent_container", basic_container_spec, "owner"
        )
        await container_manager.create_container(
            "child_container", basic_container_spec, "owner"
        )

        # Set up parent-child relationship
        parent_data = container_manager._container_registry["parent_container"]
        parent_data["child_containers"] = ["child_container"]

        results = await container_manager.search_container_contents(
            "parent_container", "item", recursive=True
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_validate_container_access_owner(
        self, container_manager, basic_container_spec
    ):
        """Test access validation for container owner."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        has_access, error = await container_manager.validate_container_access(
            "test_container", "owner", "open"
        )

        assert has_access is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_container_access_non_owner(
        self, container_manager, basic_container_spec
    ):
        """Test access validation for non-owner."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        has_access, error = await container_manager.validate_container_access(
            "test_container", "other_user", "open"
        )

        assert has_access is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_apply_container_effects_preservation(
        self, container_manager, basic_container_spec
    ):
        """Test applying preservation effects to container."""
        await container_manager.create_container(
            "magic_container", basic_container_spec, "owner"
        )

        result = await container_manager.apply_container_effects(
            "magic_container", "preservation", {"power": 2.0, "time_elapsed": 1.0}
        )

        assert result["effect_type"] == "preservation"
        assert "effects_applied" in result

    @pytest.mark.asyncio
    async def test_apply_container_effects_temperature(
        self, container_manager, basic_container_spec
    ):
        """Test applying temperature control effects."""
        await container_manager.create_container(
            "cooling_box", basic_container_spec, "owner"
        )

        result = await container_manager.apply_container_effects(
            "cooling_box", "temperature_control", {"temperature": 0.0}  # Freezing
        )

        assert result["effect_type"] == "temperature_control"

        # Check that container state was updated
        container_state = container_manager._container_states["cooling_box"]
        assert container_state["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_transfer_between_containers(
        self, container_manager, basic_container_spec
    ):
        """Test transferring items between containers."""
        await container_manager.create_container(
            "container1", basic_container_spec, "owner"
        )
        await container_manager.create_container(
            "container2", basic_container_spec, "owner"
        )

        success = await container_manager.transfer_between_containers(
            "container1", "container2", "test_item", 2
        )

        assert success is True
        # Should have called inventory manager's move_item
        container_manager.inventory.move_item.assert_called()

    @pytest.mark.asyncio
    async def test_container_access_logging(
        self, container_manager, basic_container_spec
    ):
        """Test that container access is logged."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        # Open container
        await container_manager.open_container(
            "test_container", "owner", {"test": "context"}
        )

        # Check access log
        container_data = container_manager._container_registry["test_container"]
        assert len(container_data["access_log"]) > 0

        log_entry = container_data["access_log"][0]
        assert log_entry["accessor_id"] == "owner"
        assert log_entry["action"] == "open"
        assert "timestamp" in log_entry

    @pytest.mark.asyncio
    async def test_container_special_properties_safe(self, container_manager):
        """Test container with special properties (safe type)."""
        safe_spec = ContainerSpecification(
            container_type=ContainerType.SAFE,
            capacity_slots=5,
            weight_limit=25.0,
            volume_limit=50.0,
            access_restrictions=["private"],
            organization_rules={},
            special_properties={"security_level": 5},
        )

        await container_manager.create_container("safe", safe_spec, "owner")

        # Check that safe properties were applied
        container_state = container_manager._container_states["safe"]
        assert container_state["locked"] is True
        assert container_state["security_level"] == 5

    @pytest.mark.asyncio
    async def test_container_special_properties_magical(self, container_manager):
        """Test container with magical properties."""
        magical_spec = ContainerSpecification(
            container_type=ContainerType.MAGICAL,
            capacity_slots=20,
            weight_limit=100.0,
            volume_limit=200.0,
            access_restrictions=[],
            organization_rules={},
            special_properties={"preservation": True},
        )

        await container_manager.create_container("magic_bag", magical_spec, "owner")

        # Check that magical properties were applied
        container_state = container_manager._container_states["magic_bag"]
        assert "magical_preservation" in container_state["preservation_effects"]

    @pytest.mark.asyncio
    async def test_container_organization_strategies(
        self, container_manager, basic_container_spec
    ):
        """Test different organization strategies."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        strategies = ["auto", "category", "size"]

        for strategy in strategies:
            result = await container_manager.organize_container_contents(
                "test_container", strategy
            )
            assert result["strategy"] == strategy
            assert "changes_made" in result

    @pytest.mark.asyncio
    async def test_container_metadata_tracking(
        self, container_manager, basic_container_spec
    ):
        """Test that container metadata is properly tracked."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        container_data = container_manager._container_registry["test_container"]

        assert "metadata" in container_data
        assert (
            "created_at" in container_data
        )  # created_at is in main container data, not metadata
        assert "access_count" in container_data["metadata"]
        assert container_data["metadata"]["organization_strategy"] == "auto"

    @pytest.mark.asyncio
    async def test_container_access_count_tracking(
        self, container_manager, basic_container_spec
    ):
        """Test that access count is tracked."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )
        await container_manager.open_container("test_container", "owner", {})

        # Place item (should increment access count)
        await container_manager.place_item_in_container("test_container", "item", 1)

        container_data = container_manager._container_registry["test_container"]
        assert container_data["metadata"]["access_count"] > 0

    @pytest.mark.asyncio
    async def test_container_hierarchy_depth_limit(
        self, container_manager, basic_container_spec
    ):
        """Test container hierarchy depth limiting."""
        await container_manager.create_container("root", basic_container_spec, "owner")

        # Test with limited depth
        hierarchy = await container_manager.get_container_hierarchy("root", max_depth=1)

        assert hierarchy["max_depth"] == 1
        assert "hierarchy" in hierarchy

    @pytest.mark.asyncio
    async def test_error_handling_invalid_container(self, container_manager):
        """Test error handling for operations on invalid containers."""
        success, result = await container_manager.open_container(
            "nonexistent", "user", {}
        )

        assert success is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_container_state_persistence(
        self, container_manager, basic_container_spec
    ):
        """Test that container state changes persist."""
        await container_manager.create_container(
            "test_container", basic_container_spec, "owner"
        )

        # Open container
        await container_manager.open_container("test_container", "owner", {})

        # Check state
        container_state = container_manager._container_states["test_container"]
        assert container_state["open"] is True
        assert container_state["contents_visible"] is True

        # Close container
        await container_manager.close_container("test_container", "owner")

        # Check state updated
        assert container_state["open"] is False
        assert container_state["contents_visible"] is False
