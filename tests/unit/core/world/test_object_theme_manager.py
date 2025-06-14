"""
Unit tests for ObjectThemeManager.

Tests object archetype management, theme consistency, and cultural variations.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from game_loop.core.models.object_models import (
    GeneratedObject,
    ObjectInteractions,
    ObjectProperties,
)
from game_loop.core.world.object_theme_manager import ObjectThemeManager
from game_loop.state.models import Location, WorldObject, WorldState


class TestObjectThemeManager:
    """Tests for ObjectThemeManager."""

    @pytest.fixture
    def world_state(self):
        """Create test world state."""
        return WorldState()

    @pytest.fixture
    def session_factory(self):
        """Create mock session factory."""
        return AsyncMock()

    @pytest.fixture
    def theme_manager(self, world_state, session_factory):
        """Create ObjectThemeManager instance."""
        return ObjectThemeManager(world_state, session_factory)

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

    @pytest.mark.asyncio
    async def test_get_available_object_types_village(self, theme_manager):
        """Test getting available object types for village theme."""
        object_types = await theme_manager.get_available_object_types("Village")

        assert isinstance(object_types, list)
        assert len(object_types) > 0
        assert "tool" in object_types or "container" in object_types

    @pytest.mark.asyncio
    async def test_get_available_object_types_forest(self, theme_manager):
        """Test getting available object types for forest theme."""
        object_types = await theme_manager.get_available_object_types("Forest")

        assert isinstance(object_types, list)
        assert len(object_types) > 0
        # Forest should prefer natural objects
        assert "natural" in object_types or "tool" in object_types

    @pytest.mark.asyncio
    async def test_get_available_object_types_unknown_theme(self, theme_manager):
        """Test getting available object types for unknown theme."""
        object_types = await theme_manager.get_available_object_types("UnknownTheme")

        # Should return fallback types
        assert isinstance(object_types, list)
        assert len(object_types) > 0
        assert "container" in object_types

    @pytest.mark.asyncio
    async def test_determine_object_type_populate_location(
        self, theme_manager, village_location
    ):
        """Test determining object type for populate_location purpose."""
        from game_loop.core.models.location_models import LocationTheme
        from game_loop.core.models.object_models import ObjectGenerationContext

        theme = LocationTheme(
            name="Village",
            description="Rural setting",
            visual_elements=["cobblestone", "thatched_roofs"],
            atmosphere="peaceful",
            typical_objects=["tool", "container"],
            typical_npcs=["villager", "merchant"],
            generation_parameters={},
        )
        context = ObjectGenerationContext(
            location=village_location,
            location_theme=theme,
            generation_purpose="populate_location",
            existing_objects=[],
        )

        object_type = await theme_manager.determine_object_type(context)

        assert isinstance(object_type, str)
        assert len(object_type) > 0
        # Should be a reasonable type for populating a village
        assert object_type in ["container", "tool", "furniture", "weapon"]

    @pytest.mark.asyncio
    async def test_determine_object_type_quest_related(
        self, theme_manager, village_location
    ):
        """Test determining object type for quest_related purpose."""
        from game_loop.core.models.location_models import LocationTheme
        from game_loop.core.models.object_models import ObjectGenerationContext

        theme = LocationTheme(
            name="Village",
            description="Rural setting",
            visual_elements=["cobblestone", "thatched_roofs"],
            atmosphere="peaceful",
            typical_objects=["tool", "container"],
            typical_npcs=["villager", "merchant"],
            generation_parameters={},
        )
        context = ObjectGenerationContext(
            location=village_location,
            location_theme=theme,
            generation_purpose="quest_related",
            existing_objects=[],
        )

        object_type = await theme_manager.determine_object_type(context)

        assert isinstance(object_type, str)
        # Quest objects should prefer special types, but may fall back to available types for the location
        quest_preferred = ["treasure", "key", "relic"]
        location_common = ["container", "tool", "weapon", "furniture"]
        assert object_type in quest_preferred + location_common

    @pytest.mark.asyncio
    async def test_get_object_template_sword(self, theme_manager):
        """Test getting object template for sword in village theme."""
        template = await theme_manager.get_object_template("weapon", "Village")

        assert isinstance(template, ObjectProperties)
        assert template.object_type == "weapon"
        assert template.name is not None
        assert len(template.name) > 0

    @pytest.mark.asyncio
    async def test_get_object_template_unknown_type(self, theme_manager):
        """Test getting object template for unknown type."""
        template = await theme_manager.get_object_template("unknown_type", "Village")

        assert isinstance(template, ObjectProperties)
        assert template.object_type == "unknown_type"
        assert "village" in template.name.lower()

    @pytest.mark.asyncio
    async def test_generate_cultural_variations_village(
        self, theme_manager, village_location
    ):
        """Test generating cultural variations for village theme."""
        base_properties = ObjectProperties(
            name="iron sword", object_type="weapon", material="unknown"
        )

        varied_properties = await theme_manager.generate_cultural_variations(
            base_properties, village_location
        )

        assert isinstance(varied_properties, ObjectProperties)
        assert varied_properties.object_type == base_properties.object_type
        # Material should be updated for village theme
        assert varied_properties.material != "unknown"
        # Value should be adjusted for village
        assert varied_properties.value <= base_properties.value * 1.2

    @pytest.mark.asyncio
    async def test_generate_cultural_variations_forest(
        self, theme_manager, forest_location
    ):
        """Test generating cultural variations for forest theme."""
        base_properties = ObjectProperties(
            name="wooden staff", object_type="tool", material="wood", value=100
        )

        varied_properties = await theme_manager.generate_cultural_variations(
            base_properties, forest_location
        )

        assert isinstance(varied_properties, ObjectProperties)
        # Value should be reduced for forest setting
        assert varied_properties.value < base_properties.value
        # Should have forest-appropriate special properties
        assert len(varied_properties.special_properties) >= len(
            base_properties.special_properties
        )

    @pytest.mark.asyncio
    async def test_validate_object_consistency_valid(
        self, theme_manager, village_location
    ):
        """Test validating object consistency for valid object."""
        base_object = WorldObject(
            object_id=uuid4(), name="Village Hammer", description="A sturdy hammer"
        )
        properties = ObjectProperties(
            name="Village Hammer",
            object_type="tool",
            material="iron_and_wood",
            special_properties=["practical", "sturdy"],
        )
        interactions = ObjectInteractions()

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        is_consistent = await theme_manager.validate_object_consistency(
            generated_object, village_location
        )

        assert is_consistent is True

    @pytest.mark.asyncio
    async def test_validate_object_consistency_forbidden_elements(
        self, theme_manager, village_location
    ):
        """Test validating object consistency with forbidden elements."""
        base_object = WorldObject(
            object_id=uuid4(), name="Luxury Item", description="An ornate object"
        )
        properties = ObjectProperties(
            name="Luxury Item",
            object_type="luxury",
            material="gold",
            special_properties=["luxury", "ornate"],  # Forbidden in village theme
        )
        interactions = ObjectInteractions()

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        is_consistent = await theme_manager.validate_object_consistency(
            generated_object, village_location
        )

        assert is_consistent is False

    @pytest.mark.asyncio
    async def test_validate_object_consistency_unknown_theme(self, theme_manager):
        """Test validating object consistency for unknown theme."""
        unknown_location = Location(
            location_id=uuid4(),
            name="Unknown Place",
            description="A mysterious location",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "UnknownTheme"},
        )

        base_object = WorldObject(
            object_id=uuid4(), name="Test Object", description="Test"
        )
        properties = ObjectProperties(name="Test Object", object_type="tool")
        interactions = ObjectInteractions()

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        is_consistent = await theme_manager.validate_object_consistency(
            generated_object, unknown_location
        )

        # Should default to valid for unknown themes
        assert is_consistent is True

    def test_get_archetype_definition_existing(self, theme_manager):
        """Test getting existing archetype definition."""
        archetype = theme_manager.get_archetype_definition("sword")

        assert archetype is not None
        assert archetype.name == "sword"
        assert archetype.description is not None
        assert archetype.typical_properties.object_type == "weapon"

    def test_get_archetype_definition_nonexistent(self, theme_manager):
        """Test getting non-existent archetype definition."""
        archetype = theme_manager.get_archetype_definition("nonexistent")

        assert archetype is None

    def test_get_theme_definition_existing(self, theme_manager):
        """Test getting existing theme definition."""
        theme = theme_manager.get_theme_definition("Village")

        assert theme is not None
        assert theme.name == "Village"
        assert theme.description is not None
        assert len(theme.typical_materials) > 0

    def test_get_theme_definition_nonexistent(self, theme_manager):
        """Test getting non-existent theme definition."""
        theme = theme_manager.get_theme_definition("NonexistentTheme")

        assert theme is None

    @pytest.mark.asyncio
    async def test_get_archetype_for_object_type_weapon(self, theme_manager):
        """Test getting archetype for weapon object type."""
        archetype_name = await theme_manager.get_archetype_for_object_type(
            "weapon", "Village"
        )

        assert archetype_name is not None
        assert isinstance(archetype_name, str)
        # Should be an archetype that creates weapons
        archetype = theme_manager.get_archetype_definition(archetype_name)
        assert archetype.typical_properties.object_type == "weapon"

    @pytest.mark.asyncio
    async def test_get_archetype_for_object_type_nonexistent(self, theme_manager):
        """Test getting archetype for non-existent object type."""
        archetype_name = await theme_manager.get_archetype_for_object_type(
            "nonexistent_type", "Village"
        )

        assert archetype_name is None

    def test_initialization_creates_default_archetypes(self, theme_manager):
        """Test that initialization creates default archetypes."""
        # Check that common archetypes exist
        sword_archetype = theme_manager.get_archetype_definition("sword")
        hammer_archetype = theme_manager.get_archetype_definition("hammer")
        chest_archetype = theme_manager.get_archetype_definition("chest")

        assert sword_archetype is not None
        assert hammer_archetype is not None
        assert chest_archetype is not None

        # Check their properties
        assert sword_archetype.typical_properties.object_type == "weapon"
        assert hammer_archetype.typical_properties.object_type == "tool"
        assert chest_archetype.typical_properties.object_type == "container"

    def test_initialization_creates_default_themes(self, theme_manager):
        """Test that initialization creates default themes."""
        # Check that common themes exist
        village_theme = theme_manager.get_theme_definition("Village")
        forest_theme = theme_manager.get_theme_definition("Forest")
        city_theme = theme_manager.get_theme_definition("City")
        dungeon_theme = theme_manager.get_theme_definition("Dungeon")

        assert village_theme is not None
        assert forest_theme is not None
        assert city_theme is not None
        assert dungeon_theme is not None

        # Check their properties
        assert "wood" in village_theme.typical_materials
        assert "natural" in forest_theme.style_descriptors
        assert "sophisticated" in city_theme.style_descriptors
        assert "ancient" in dungeon_theme.style_descriptors

    @pytest.mark.asyncio
    async def test_archetype_location_affinities(self, theme_manager):
        """Test that archetypes have appropriate location affinities."""
        sword_archetype = theme_manager.get_archetype_definition("sword")
        herb_archetype = theme_manager.get_archetype_definition("herb")

        # Sword should have higher affinity for combat locations
        assert sword_archetype.location_affinities.get("City", 0) > 0.5

        # Herb should have higher affinity for natural locations
        assert herb_archetype.location_affinities.get("Forest", 0) > 0.8

    @pytest.mark.asyncio
    async def test_error_handling_in_determine_object_type(self, theme_manager):
        """Test error handling in determine_object_type."""
        # Create invalid context that might cause errors
        invalid_location = Location(
            location_id=uuid4(),
            name="Invalid",
            description="",
            connections={},
            objects={},
            npcs={},
            state_flags={},  # Missing theme
        )

        from game_loop.core.models.location_models import LocationTheme
        from game_loop.core.models.object_models import ObjectGenerationContext

        theme = LocationTheme(
            name="Unknown",
            description="",
            visual_elements=[],
            atmosphere="neutral",
            typical_objects=[],
            typical_npcs=[],
            generation_parameters={},
        )
        context = ObjectGenerationContext(
            location=invalid_location,
            location_theme=theme,
            generation_purpose="populate_location",
            existing_objects=[],
        )

        # Should handle gracefully and return fallback
        object_type = await theme_manager.determine_object_type(context)
        assert object_type == "container"  # Fallback type
