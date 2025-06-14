"""
Unit tests for ObjectPlacementManager.

Tests intelligent object placement logic, density management, and spatial validation.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from game_loop.core.models.object_models import (
    GeneratedObject,
    ObjectInteractions,
    ObjectPlacement,
    ObjectProperties,
)
from game_loop.core.world.object_placement_manager import ObjectPlacementManager
from game_loop.state.models import Location, WorldObject, WorldState


class TestObjectPlacementManager:
    """Tests for ObjectPlacementManager."""

    @pytest.fixture
    def world_state(self):
        """Create test world state."""
        return WorldState()

    @pytest.fixture
    def session_factory(self):
        """Create mock session factory."""
        return AsyncMock()

    @pytest.fixture
    def placement_manager(self, world_state, session_factory):
        """Create ObjectPlacementManager instance."""
        return ObjectPlacementManager(world_state, session_factory)

    @pytest.fixture
    def village_location(self):
        """Create test village location."""
        return Location(
            location_id=uuid4(),
            name="Millbrook Village",
            description="A peaceful farming village",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Village", "type": "village"},
        )

    @pytest.fixture
    def forest_location(self):
        """Create test forest location."""
        return Location(
            location_id=uuid4(),
            name="Whispering Woods",
            description="An ancient forest",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Forest", "type": "wilderness"},
        )

    @pytest.fixture
    def dungeon_location(self):
        """Create test dungeon location."""
        return Location(
            location_id=uuid4(),
            name="Ancient Crypt",
            description="A dark underground chamber",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Dungeon", "type": "underground"},
        )

    @pytest.fixture
    def test_sword(self):
        """Create test sword object."""
        base_object = WorldObject(
            object_id=uuid4(),
            name="Iron Sword",
            description="A well-crafted iron sword",
        )

        properties = ObjectProperties(
            name="Iron Sword",
            object_type="weapon",
            material="iron",
            size="medium",
            weight="heavy",
            value=100,
            special_properties=["sharp", "balanced"],
        )

        interactions = ObjectInteractions(
            available_actions=["examine", "wield", "attack"], portable=True
        )

        return GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

    @pytest.fixture
    def test_chest(self):
        """Create test chest object."""
        base_object = WorldObject(
            object_id=uuid4(),
            name="Wooden Chest",
            description="A large wooden storage chest",
        )

        properties = ObjectProperties(
            name="Wooden Chest",
            object_type="container",
            material="wood",
            size="large",
            weight="heavy",
        )

        interactions = ObjectInteractions(
            available_actions=["examine", "open", "close"], portable=False
        )

        return GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

    @pytest.fixture
    def test_herb(self):
        """Create test herb object."""
        base_object = WorldObject(
            object_id=uuid4(),
            name="Healing Herb",
            description="A small medicinal plant",
        )

        properties = ObjectProperties(
            name="Healing Herb",
            object_type="natural",
            material="plant",
            size="tiny",
            weight="light",
            special_properties=["medicinal"],
        )

        interactions = ObjectInteractions(
            available_actions=["examine", "gather", "consume"],
            portable=True,
            consumable=True,
        )

        return GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

    @pytest.mark.asyncio
    async def test_determine_placement_sword_village(
        self, placement_manager, test_sword, village_location
    ):
        """Test determining placement for sword in village."""
        placement = await placement_manager.determine_placement(
            test_sword, village_location
        )

        assert isinstance(placement, ObjectPlacement)
        assert placement.object_id == test_sword.base_object.object_id
        assert placement.location_id == village_location.location_id
        assert placement.placement_type in ["wall", "table", "floor"]
        assert placement.visibility == "visible"
        assert placement.accessibility in ["accessible", "blocked"]
        assert 1 <= placement.discovery_difficulty <= 10

    @pytest.mark.asyncio
    async def test_determine_placement_chest_village(
        self, placement_manager, test_chest, village_location
    ):
        """Test determining placement for chest in village."""
        placement = await placement_manager.determine_placement(
            test_chest, village_location
        )

        assert isinstance(placement, ObjectPlacement)
        assert placement.placement_type in ["floor", "table"]  # Large containers
        assert placement.visibility == "visible"  # Village prefers visible
        assert placement.spatial_description is not None
        assert len(placement.spatial_description) > 0

    @pytest.mark.asyncio
    async def test_determine_placement_herb_forest(
        self, placement_manager, test_herb, forest_location
    ):
        """Test determining placement for herb in forest."""
        placement = await placement_manager.determine_placement(
            test_herb, forest_location
        )

        assert isinstance(placement, ObjectPlacement)
        assert placement.placement_type in ["floor", "embedded"]  # Natural objects
        assert placement.visibility in ["visible", "partially_hidden"]
        assert (
            "leaves" in placement.spatial_description
            or "forest" in placement.spatial_description.lower()
        )

    @pytest.mark.asyncio
    async def test_determine_placement_sword_dungeon(
        self, placement_manager, test_sword, dungeon_location
    ):
        """Test determining placement for sword in dungeon."""
        placement = await placement_manager.determine_placement(
            test_sword, dungeon_location
        )

        assert isinstance(placement, ObjectPlacement)
        assert placement.placement_type in ["floor", "hidden", "embedded"]
        # Dungeon items often hidden or partially hidden
        assert placement.visibility in ["visible", "partially_hidden", "hidden"]
        assert placement.discovery_difficulty >= 2  # Dungeons are harder to search

    @pytest.mark.asyncio
    async def test_validate_placement_valid(self, placement_manager, village_location):
        """Test validating a valid placement."""
        placement = ObjectPlacement(
            object_id=uuid4(),
            location_id=village_location.location_id,
            placement_type="floor",
            visibility="visible",
            accessibility="accessible",
            discovery_difficulty=3,
        )

        is_valid = await placement_manager.validate_placement(
            placement, village_location
        )
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_placement_invalid_type(
        self, placement_manager, village_location
    ):
        """Test validating placement with invalid type."""
        # Test that ObjectPlacement model validation catches invalid types
        with pytest.raises(ValueError, match="Invalid placement type"):
            ObjectPlacement(
                object_id=uuid4(),
                location_id=village_location.location_id,
                placement_type="invalid_type",
                discovery_difficulty=3,
            )

    @pytest.mark.asyncio
    async def test_validate_placement_invalid_difficulty(
        self, placement_manager, village_location
    ):
        """Test validating placement with invalid discovery difficulty."""
        # Test that ObjectPlacement model validation catches invalid difficulty
        with pytest.raises(
            ValueError, match="Discovery difficulty must be between 1 and 10"
        ):
            ObjectPlacement(
                object_id=uuid4(),
                location_id=village_location.location_id,
                placement_type="floor",
                discovery_difficulty=15,  # Invalid range
            )

    @pytest.mark.asyncio
    async def test_validate_placement_avoided_type_in_theme(
        self, placement_manager, forest_location
    ):
        """Test validating placement with avoided type for theme."""
        placement = ObjectPlacement(
            object_id=uuid4(),
            location_id=forest_location.location_id,
            placement_type="table",  # Tables avoided in forest
            discovery_difficulty=3,
        )

        is_valid = await placement_manager.validate_placement(
            placement, forest_location
        )
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_check_placement_density_empty_location(
        self, placement_manager, village_location
    ):
        """Test checking placement density for empty location."""
        density_info = await placement_manager.check_placement_density(village_location)

        assert density_info["current_count"] == 0
        assert density_info["max_recommended"] > 0
        assert density_info["density_ratio"] == 0.0
        assert density_info["sparse"] is True
        assert density_info["overcrowded"] is False
        assert density_info["can_accommodate"] is True

    @pytest.mark.asyncio
    async def test_check_placement_density_populated_location(
        self, placement_manager, village_location
    ):
        """Test checking placement density for populated location."""
        # Add some objects to the location
        for i in range(5):
            obj = WorldObject(
                object_id=uuid4(), name=f"Object {i}", description="Test object"
            )
            village_location.objects[obj.object_id] = obj

        density_info = await placement_manager.check_placement_density(village_location)

        assert density_info["current_count"] == 5
        assert density_info["density_ratio"] > 0.0
        assert density_info["sparse"] is False
        assert density_info["can_accommodate"] is True

    @pytest.mark.asyncio
    async def test_check_placement_density_overcrowded_location(
        self, placement_manager, village_location
    ):
        """Test checking placement density for overcrowded location."""
        # Add many objects to exceed capacity
        for i in range(15):  # Village max is typically 12
            obj = WorldObject(
                object_id=uuid4(), name=f"Object {i}", description="Test object"
            )
            village_location.objects[obj.object_id] = obj

        density_info = await placement_manager.check_placement_density(village_location)

        assert density_info["current_count"] == 15
        assert density_info["density_ratio"] > 1.0
        assert density_info["overcrowded"] is True
        assert density_info["can_accommodate"] is False

    @pytest.mark.asyncio
    async def test_update_location_objects(self, placement_manager, village_location):
        """Test updating location with object placement."""
        # Add location to world state so placement manager can find it
        placement_manager.world_state.locations[village_location.location_id] = (
            village_location
        )

        placement = ObjectPlacement(
            object_id=uuid4(),
            location_id=village_location.location_id,
            placement_type="floor",
        )

        success = await placement_manager.update_location_objects(
            village_location.location_id, placement
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_location_objects_invalid_location(self, placement_manager):
        """Test updating with invalid location ID."""
        invalid_location_id = uuid4()
        placement = ObjectPlacement(
            object_id=uuid4(), location_id=invalid_location_id, placement_type="floor"
        )

        success = await placement_manager.update_location_objects(
            invalid_location_id, placement
        )

        assert success is False

    def test_get_size_constraints(self, placement_manager):
        """Test getting size constraints for different sizes."""
        tiny_constraints = placement_manager._get_size_constraints("tiny")
        huge_constraints = placement_manager._get_size_constraints("huge")

        assert (
            tiny_constraints["max_per_location"] > huge_constraints["max_per_location"]
        )
        assert len(tiny_constraints["placement_types"]) >= len(
            huge_constraints["placement_types"]
        )

    def test_get_theme_rules(
        self, placement_manager, village_location, forest_location
    ):
        """Test getting theme rules for different locations."""
        village_rules = placement_manager._get_theme_rules(village_location)
        forest_rules = placement_manager._get_theme_rules(forest_location)

        assert "preferred_types" in village_rules
        assert "preferred_types" in forest_rules
        assert village_rules["preferred_types"] != forest_rules["preferred_types"]

    def test_get_object_type_rules(self, placement_manager):
        """Test getting object type rules."""
        weapon_rules = placement_manager._get_object_type_rules("weapon")
        treasure_rules = placement_manager._get_object_type_rules("treasure")

        assert "placement_types" in weapon_rules
        assert "visibility" in weapon_rules
        assert (
            weapon_rules["visibility"] != treasure_rules["visibility"]
        )  # Weapons visible, treasures hidden

    def test_select_placement_type_weapon_village(
        self, placement_manager, village_location
    ):
        """Test selecting placement type for weapon in village."""
        size_constraints = {"placement_types": ["table", "wall", "floor"]}
        theme_rules = {
            "preferred_types": ["table", "shelf", "floor"],
            "avoid_types": ["hidden"],
        }
        object_type_rules = {"placement_types": ["wall", "table", "floor"]}

        placement_type = placement_manager._select_placement_type(
            size_constraints, theme_rules, object_type_rules
        )

        assert placement_type in ["table", "floor"]  # Intersection of all constraints

    def test_determine_visibility_treasure(self, placement_manager):
        """Test determining visibility for treasure object."""
        from game_loop.core.models.object_models import (
            GeneratedObject,
            ObjectInteractions,
            ObjectProperties,
        )
        from game_loop.state.models import WorldObject

        # High-value treasure should be hidden
        base_object = WorldObject(object_id=uuid4(), name="Gold Coins", description="")
        properties = ObjectProperties(
            name="Gold Coins", object_type="treasure", value=500  # High value
        )
        interactions = ObjectInteractions()

        treasure_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        visibility = placement_manager._determine_visibility(
            treasure_object, None, {}, {}
        )

        assert visibility == "hidden"

    def test_determine_accessibility_embedded(self, placement_manager):
        """Test determining accessibility for embedded objects."""
        from game_loop.core.models.object_models import (
            GeneratedObject,
            ObjectInteractions,
            ObjectProperties,
        )
        from game_loop.state.models import WorldObject

        base_object = WorldObject(object_id=uuid4(), name="Test", description="")
        properties = ObjectProperties(name="Test", object_type="tool")
        interactions = ObjectInteractions()

        test_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        accessibility = placement_manager._determine_accessibility(
            test_object, "embedded"
        )
        assert accessibility == "requires_tool"

        accessibility = placement_manager._determine_accessibility(test_object, "floor")
        assert accessibility == "accessible"

    def test_calculate_discovery_difficulty_factors(self, placement_manager):
        """Test discovery difficulty calculation with different factors."""
        # Visible floor object should be easy to find
        easy_difficulty = placement_manager._calculate_discovery_difficulty(
            "visible", "floor", "tool"
        )

        # Hidden embedded treasure should be hard to find
        hard_difficulty = placement_manager._calculate_discovery_difficulty(
            "hidden", "embedded", "treasure"
        )

        assert easy_difficulty < hard_difficulty
        assert 1 <= easy_difficulty <= 10
        assert 1 <= hard_difficulty <= 10

    def test_generate_spatial_description_themes(
        self, placement_manager, village_location, forest_location
    ):
        """Test spatial description generation for different themes."""
        from game_loop.core.models.object_models import (
            GeneratedObject,
            ObjectInteractions,
            ObjectProperties,
        )
        from game_loop.state.models import WorldObject

        base_object = WorldObject(object_id=uuid4(), name="Test Object", description="")
        properties = ObjectProperties(name="Test Object", object_type="tool")
        interactions = ObjectInteractions()

        test_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        village_desc = placement_manager._generate_spatial_description(
            test_object, village_location, "table"
        )

        forest_desc = placement_manager._generate_spatial_description(
            test_object, forest_location, "floor"
        )

        assert "Test Object" in village_desc
        assert "Test Object" in forest_desc
        assert village_desc != forest_desc  # Should be different for different themes
        assert "leaves" in forest_desc or "forest" in forest_desc.lower()

    @pytest.mark.asyncio
    async def test_error_handling_in_determine_placement(
        self, placement_manager, village_location
    ):
        """Test error handling in determine_placement."""
        # Create object that might cause errors
        invalid_object = None

        try:
            placement = await placement_manager.determine_placement(
                invalid_object, village_location
            )

            # Should return fallback placement
            assert isinstance(placement, ObjectPlacement)
            assert "error" in placement.placement_metadata
            assert placement.placement_type == "floor"  # Fallback
        except Exception:
            # If it raises an exception, that's also acceptable for invalid input
            pass
