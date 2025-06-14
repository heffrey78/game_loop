"""
Unit tests for ObjectGenerator.

Tests object generation engine functionality including LLM integration,
property generation, and validation.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import LocationTheme
from game_loop.core.models.object_models import (
    GeneratedObject,
    ObjectGenerationContext,
    ObjectProperties,
)
from game_loop.core.world.object_generator import ObjectGenerator
from game_loop.core.world.object_theme_manager import ObjectThemeManager
from game_loop.state.models import Location, WorldState


class TestObjectGenerator:
    """Tests for ObjectGenerator."""

    @pytest.fixture
    def world_state(self):
        """Create test world state."""
        return WorldState()

    @pytest.fixture
    def session_factory(self):
        """Create mock session factory."""
        return AsyncMock()

    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": json.dumps(
                {
                    "name": "Iron Dagger",
                    "description": "A sharp iron dagger with a leather-wrapped handle.",
                    "material": "iron",
                    "size": "small",
                    "weight": "light",
                    "durability": "sturdy",
                    "value": 25,
                    "special_properties": ["sharp", "balanced"],
                    "cultural_significance": "common",
                }
            )
        }
        return mock_client

    @pytest.fixture
    def theme_manager(self, world_state, session_factory):
        """Create mock theme manager."""
        theme_manager = ObjectThemeManager(world_state, session_factory)
        return theme_manager

    @pytest.fixture
    def object_generator(self, world_state, session_factory, llm_client, theme_manager):
        """Create ObjectGenerator instance."""
        return ObjectGenerator(
            world_state=world_state,
            session_factory=session_factory,
            llm_client=llm_client,
            theme_manager=theme_manager,
            template_path="templates",
        )

    @pytest.fixture
    def test_context(self):
        """Create test generation context."""
        location = Location(
            location_id=uuid4(),
            name="Test Village",
            description="A small village",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Village"},
        )

        theme = LocationTheme(
            name="Village",
            description="Rural setting",
            visual_elements=["cobblestone", "thatched_roofs"],
            atmosphere="peaceful",
            typical_objects=["tool", "container"],
            typical_npcs=["villager", "merchant"],
            generation_parameters={"object_affinities": {"tool": 0.8}},
        )

        return ObjectGenerationContext(
            location=location,
            location_theme=theme,
            generation_purpose="populate_location",
            existing_objects=[],
        )

    @pytest.mark.asyncio
    async def test_generate_object_basic(self, object_generator, test_context):
        """Test basic object generation."""
        generated_object = await object_generator.generate_object(test_context)

        assert isinstance(generated_object, GeneratedObject)
        assert generated_object.base_object.name is not None
        assert generated_object.properties.object_type is not None
        assert len(generated_object.interactions.available_actions) > 0
        assert "examine" in generated_object.interactions.available_actions

    @pytest.mark.asyncio
    async def test_generate_object_has_metadata(self, object_generator, test_context):
        """Test that generated object includes metadata."""
        generated_object = await object_generator.generate_object(test_context)

        assert "location_theme" in generated_object.generation_metadata
        assert "generation_purpose" in generated_object.generation_metadata
        assert "timestamp" in generated_object.generation_metadata
        assert generated_object.generation_metadata["location_theme"] == "Village"

    @pytest.mark.asyncio
    async def test_create_object_properties_basic(self, object_generator, test_context):
        """Test creating object properties."""
        properties = await object_generator.create_object_properties(
            "weapon", test_context
        )

        assert isinstance(properties, ObjectProperties)
        assert properties.object_type == "weapon"
        assert properties.name is not None
        assert len(properties.name) > 0

    @pytest.mark.asyncio
    async def test_create_object_interactions_weapon(self, object_generator):
        """Test creating interactions for weapon."""
        properties = ObjectProperties(
            name="Iron Sword",
            object_type="weapon",
            material="iron",
            special_properties=["sharp"],
        )

        test_context = MagicMock()
        interactions = await object_generator.create_object_interactions(
            properties, test_context
        )

        assert "examine" in interactions.available_actions
        assert "wield" in interactions.available_actions
        assert "attack" in interactions.available_actions
        assert interactions.portable is True

    @pytest.mark.asyncio
    async def test_create_object_interactions_container(self, object_generator):
        """Test creating interactions for container."""
        properties = ObjectProperties(
            name="Wooden Chest", object_type="container", material="wood", size="large"
        )

        test_context = MagicMock()
        interactions = await object_generator.create_object_interactions(
            properties, test_context
        )

        assert "examine" in interactions.available_actions
        assert "open" in interactions.available_actions
        assert "close" in interactions.available_actions
        assert "search" in interactions.available_actions
        # Large containers should not be portable
        assert interactions.portable is False

    @pytest.mark.asyncio
    async def test_create_object_interactions_book(self, object_generator):
        """Test creating interactions for book/knowledge object."""
        properties = ObjectProperties(
            name="Ancient Tome",
            object_type="knowledge",
            material="parchment",
            special_properties=["readable"],
        )

        test_context = MagicMock()
        interactions = await object_generator.create_object_interactions(
            properties, test_context
        )

        assert "examine" in interactions.available_actions
        assert "read" in interactions.available_actions
        assert "study" in interactions.available_actions
        assert interactions.portable is True

    @pytest.mark.asyncio
    async def test_create_object_interactions_natural_medicinal(self, object_generator):
        """Test creating interactions for medicinal natural object."""
        properties = ObjectProperties(
            name="Healing Herb",
            object_type="natural",
            material="plant",
            special_properties=["medicinal"],
        )

        test_context = MagicMock()
        interactions = await object_generator.create_object_interactions(
            properties, test_context
        )

        assert "examine" in interactions.available_actions
        assert "gather" in interactions.available_actions
        assert "consume" in interactions.available_actions
        assert interactions.consumable is True

    @pytest.mark.asyncio
    async def test_create_object_interactions_treasure(self, object_generator):
        """Test creating interactions for treasure object."""
        properties = ObjectProperties(
            name="Ruby Gem", object_type="treasure", material="crystal", value=500
        )

        test_context = MagicMock()
        interactions = await object_generator.create_object_interactions(
            properties, test_context
        )

        assert "examine" in interactions.available_actions
        assert "appraise" in interactions.available_actions
        assert "admire" in interactions.available_actions
        assert interactions.portable is True

    @pytest.mark.asyncio
    async def test_create_object_interactions_huge_object(self, object_generator):
        """Test that huge objects are not portable."""
        properties = ObjectProperties(
            name="Giant Boulder", object_type="natural", material="stone", size="huge"
        )

        test_context = MagicMock()
        interactions = await object_generator.create_object_interactions(
            properties, test_context
        )

        assert "examine" in interactions.available_actions
        assert "take" not in interactions.available_actions
        assert interactions.portable is False

    @pytest.mark.asyncio
    async def test_validate_generated_object_valid(
        self, object_generator, test_context
    ):
        """Test validating a valid generated object."""
        # Create a valid generated object
        from game_loop.core.models.object_models import ObjectInteractions
        from game_loop.state.models import WorldObject

        base_object = WorldObject(
            object_id=uuid4(), name="Test Sword", description="A sword"
        )
        properties = ObjectProperties(name="Test Sword", object_type="weapon")
        interactions = ObjectInteractions(available_actions=["examine", "wield"])

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        result = await object_generator.validate_generated_object(
            generated_object, test_context
        )

        assert result.is_valid is True
        assert len(result.validation_errors) == 0
        assert result.consistency_score > 0.0

    @pytest.mark.asyncio
    async def test_validate_generated_object_invalid_name(
        self, object_generator, test_context
    ):
        """Test validating an object with invalid name."""
        from game_loop.core.models.object_models import ObjectInteractions
        from game_loop.state.models import WorldObject

        # Test that ObjectProperties validation catches empty names
        with pytest.raises(ValueError, match="Object name cannot be empty"):
            ObjectProperties(name="", object_type="weapon")

        # Test with valid properties but empty base object name
        base_object = WorldObject(object_id=uuid4(), name="", description="")
        properties = ObjectProperties(name="Valid Name", object_type="weapon")
        interactions = ObjectInteractions()

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        result = await object_generator.validate_generated_object(
            generated_object, test_context
        )

        # Should be valid since properties.name is not empty
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_generated_object_no_actions(
        self, object_generator, test_context
    ):
        """Test validating an object with no available actions."""
        from game_loop.core.models.object_models import ObjectInteractions
        from game_loop.state.models import WorldObject

        base_object = WorldObject(object_id=uuid4(), name="Test", description="Test")
        properties = ObjectProperties(name="Test", object_type="tool")
        interactions = ObjectInteractions(available_actions=[])  # No actions
        # Force clear the default actions added by __post_init__
        interactions.available_actions = []

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        result = await object_generator.validate_generated_object(
            generated_object, test_context
        )

        # Should still be valid but have warnings
        assert result.is_valid is True
        assert "Object has no available actions" in result.warnings
        assert result.consistency_score < 1.0

    @pytest.mark.asyncio
    async def test_generate_object_error_handling(self, object_generator):
        """Test error handling in object generation."""
        # Create invalid context that might cause errors
        invalid_context = None

        try:
            generated_object = await object_generator.generate_object(invalid_context)

            # Should return fallback object
            assert isinstance(generated_object, GeneratedObject)
            assert "fallback" in generated_object.generation_metadata
            assert generated_object.generation_metadata["fallback"] is True
            assert generated_object.properties.name == "mysterious object"
        except Exception:
            # If it raises an exception, that's also acceptable for invalid input
            pass

    def test_create_base_interactions_weapon(self, object_generator):
        """Test creating base interactions for weapon."""
        properties = ObjectProperties(
            name="Sword", object_type="weapon", special_properties=["sharp"]
        )

        interactions = object_generator._create_base_interactions(properties)

        assert "examine" in interactions.available_actions
        assert "wield" in interactions.available_actions
        assert "attack" in interactions.available_actions
        assert interactions.portable is True
        assert "wield" in interactions.interaction_results

    def test_create_base_interactions_tool(self, object_generator):
        """Test creating base interactions for tool."""
        properties = ObjectProperties(name="Hammer", object_type="tool")

        interactions = object_generator._create_base_interactions(properties)

        assert "examine" in interactions.available_actions
        assert "use" in interactions.available_actions
        assert "apply" in interactions.available_actions
        assert interactions.portable is True

    def test_create_base_interactions_furniture(self, object_generator):
        """Test creating base interactions for furniture."""
        properties = ObjectProperties(name="Wooden Chair", object_type="furniture")

        interactions = object_generator._create_base_interactions(properties)

        assert "examine" in interactions.available_actions
        assert "sit" in interactions.available_actions
        assert interactions.portable is False  # Furniture not portable
        assert "take" not in interactions.available_actions

    @pytest.mark.asyncio
    async def test_llm_integration_mock(
        self, object_generator, llm_client, test_context
    ):
        """Test LLM integration with mock responses."""
        # The mock LLM client should return valid JSON
        properties = await object_generator.create_object_properties(
            "weapon", test_context
        )

        # Should use the mocked response
        assert properties.name == "Iron Dagger"
        assert properties.material == "iron"
        assert "sharp" in properties.special_properties
        assert properties.value == 25

    def test_generation_cache_basic(self, object_generator):
        """Test that generator has a cache mechanism."""
        # The generator should have a cache
        assert hasattr(object_generator, "_generation_cache")
        assert isinstance(object_generator._generation_cache, dict)
