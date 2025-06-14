"""
Integration tests for the complete object generation pipeline.

Tests the full workflow from context collection through object generation,
storage, and placement.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from game_loop.core.models.object_models import ObjectGenerationContext
from game_loop.core.world.object_context_collector import ObjectContextCollector
from game_loop.core.world.object_generator import ObjectGenerator
from game_loop.core.world.object_placement_manager import ObjectPlacementManager
from game_loop.core.world.object_storage import ObjectStorage
from game_loop.core.world.object_theme_manager import ObjectThemeManager
from game_loop.state.models import Location, WorldState


class TestObjectGenerationPipeline:
    """Integration tests for the complete object generation pipeline."""

    @pytest.fixture
    def world_state(self):
        """Create test world state with locations."""
        world_state = WorldState()

        # Add test village
        village_id = uuid4()
        village = Location(
            location_id=village_id,
            name="Millbrook Village",
            description="A peaceful farming village nestled in a valley",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Village", "type": "village", "population": "small"},
        )
        world_state.locations[village_id] = village

        # Add test forest
        forest_id = uuid4()
        forest = Location(
            location_id=forest_id,
            name="Whispering Woods",
            description="An ancient forest with mysterious sounds",
            connections={"south": village_id},
            objects={},
            npcs={},
            state_flags={
                "theme": "Forest",
                "type": "wilderness",
                "danger_level": "low",
            },
        )
        world_state.locations[forest_id] = forest

        return world_state

    @pytest.fixture
    def session_factory(self):
        """Create mock session factory."""
        session_factory = AsyncMock()

        # Mock session context manager
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=session)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        session_factory.get_session.return_value = context_manager

        return session_factory

    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client."""
        import json

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": json.dumps(
                {
                    "name": "Village Hammer",
                    "description": "A well-used hammer with a worn wooden handle, clearly crafted by local hands.",
                    "material": "iron_and_wood",
                    "size": "medium",
                    "weight": "heavy",
                    "durability": "sturdy",
                    "value": 25,
                    "special_properties": ["practical", "well-used", "reliable"],
                    "cultural_significance": "common",
                }
            )
        }
        return mock_client

    @pytest.fixture
    def embedding_manager(self):
        """Create mock embedding manager."""
        mock_manager = AsyncMock()
        mock_manager.generate_embedding.return_value = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ] * 307  # 1535 elements
        mock_manager.generate_embedding.return_value.append(0.6)  # Make it 1536
        return mock_manager

    @pytest.fixture
    def object_pipeline(
        self, world_state, session_factory, llm_client, embedding_manager
    ):
        """Create complete object generation pipeline."""
        theme_manager = ObjectThemeManager(world_state, session_factory)
        context_collector = ObjectContextCollector(world_state, session_factory)
        object_generator = ObjectGenerator(
            world_state, session_factory, llm_client, theme_manager, "templates"
        )
        object_storage = ObjectStorage(world_state, session_factory, embedding_manager)
        placement_manager = ObjectPlacementManager(world_state, session_factory)

        return {
            "theme_manager": theme_manager,
            "context_collector": context_collector,
            "object_generator": object_generator,
            "object_storage": object_storage,
            "placement_manager": placement_manager,
        }

    @pytest.mark.asyncio
    async def test_complete_object_generation_pipeline(
        self, world_state, object_pipeline
    ):
        """Test the complete object generation pipeline."""
        # Get components
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]
        placement_manager = object_pipeline["placement_manager"]

        # Get a location to generate objects for
        village_id = list(world_state.locations.keys())[0]

        # Step 1: Collect generation context
        context = await context_collector.collect_generation_context(
            village_id, "populate_location"
        )

        assert isinstance(context, ObjectGenerationContext)
        assert context.location.name == "Millbrook Village"
        assert context.location_theme.name == "Village"
        assert context.generation_purpose == "populate_location"

        # Step 2: Generate object
        generated_object = await object_generator.generate_object(context)

        assert generated_object is not None
        assert generated_object.base_object.name is not None
        assert generated_object.properties.object_type is not None
        assert len(generated_object.interactions.available_actions) > 0

        # Step 3: Determine placement
        placement = await placement_manager.determine_placement(
            generated_object, context.location
        )

        assert placement is not None
        assert placement.object_id == generated_object.base_object.object_id
        assert placement.location_id == village_id
        assert placement.placement_type is not None

        # Step 4: Validate placement
        is_valid_placement = await placement_manager.validate_placement(
            placement, context.location
        )

        assert is_valid_placement is True

    @pytest.mark.asyncio
    async def test_object_generation_different_purposes(
        self, world_state, object_pipeline
    ):
        """Test object generation for different purposes."""
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]

        village_id = list(world_state.locations.keys())[0]

        purposes = ["populate_location", "quest_related", "random_encounter"]

        for purpose in purposes:
            # Collect context for this purpose
            context = await context_collector.collect_generation_context(
                village_id, purpose
            )

            assert context.generation_purpose == purpose

            # Generate object
            generated_object = await object_generator.generate_object(context)

            assert generated_object is not None
            assert purpose in generated_object.generation_metadata.get(
                "generation_purpose", ""
            )

    @pytest.mark.asyncio
    async def test_object_generation_different_themes(
        self, world_state, object_pipeline
    ):
        """Test object generation for different location themes."""
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]

        # Test both village and forest
        for location_id in list(world_state.locations.keys())[:2]:
            location = world_state.locations[location_id]

            # Collect context
            context = await context_collector.collect_generation_context(
                location_id, "populate_location"
            )

            # Generate object
            generated_object = await object_generator.generate_object(context)

            assert generated_object is not None

            # Object should reflect the theme
            theme_name = location.state_flags.get("theme", "Unknown")
            assert context.location_theme.name == theme_name
            assert (
                generated_object.generation_metadata.get("location_theme") == theme_name
            )

    @pytest.mark.asyncio
    async def test_object_validation_pipeline(self, world_state, object_pipeline):
        """Test object validation throughout the pipeline."""
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]
        theme_manager = object_pipeline["theme_manager"]

        village_id = list(world_state.locations.keys())[0]
        village = world_state.locations[village_id]

        # Generate object
        context = await context_collector.collect_generation_context(
            village_id, "populate_location"
        )
        generated_object = await object_generator.generate_object(context)

        # Validate object consistency
        is_consistent = await theme_manager.validate_object_consistency(
            generated_object, village
        )

        assert is_consistent is True

        # Validate object generation itself
        validation_result = await object_generator.validate_generated_object(
            generated_object, context
        )

        assert validation_result.is_valid is True
        assert len(validation_result.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_density_management_pipeline(self, world_state, object_pipeline):
        """Test density management in the pipeline."""
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]
        placement_manager = object_pipeline["placement_manager"]

        village_id = list(world_state.locations.keys())[0]
        village = world_state.locations[village_id]

        # Check initial density
        initial_density = await placement_manager.check_placement_density(village)
        assert initial_density["current_count"] == 0
        assert initial_density["can_accommodate"] is True

        # Generate multiple objects
        generated_objects = []
        for i in range(3):
            context = await context_collector.collect_generation_context(
                village_id, "populate_location"
            )

            # Add previous objects to context
            context.existing_objects = [obj.base_object for obj in generated_objects]

            generated_object = await object_generator.generate_object(context)
            generated_objects.append(generated_object)

            # Add to location for density tracking
            village.objects[generated_object.base_object.object_id] = (
                generated_object.base_object
            )

        # Check updated density
        updated_density = await placement_manager.check_placement_density(village)
        assert updated_density["current_count"] == 3
        assert updated_density["density_ratio"] > initial_density["density_ratio"]

    @pytest.mark.asyncio
    async def test_context_collection_analysis(self, world_state, object_pipeline):
        """Test context collection and analysis."""
        context_collector = object_pipeline["context_collector"]

        village_id = list(world_state.locations.keys())[0]
        village = world_state.locations[village_id]

        # Test location needs analysis
        location_needs = await context_collector.analyze_location_needs(village)

        assert "missing_object_types" in location_needs
        assert "density_level" in location_needs
        assert "thematic_needs" in location_needs
        assert location_needs["density_level"] == "sparse"  # Empty location

        # Test object context gathering
        object_context = await context_collector.gather_object_context(village_id)

        assert "density" in object_context
        assert "object_types" in object_context
        assert object_context["density"] == "sparse"  # Empty location

        # Test world knowledge collection
        world_knowledge = await context_collector.collect_world_knowledge(village)

        assert "economic_status" in world_knowledge
        assert "cultural_influences" in world_knowledge
        # Village should have modest economic status
        assert world_knowledge["economic_status"] in ["normal", "modest"]

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, world_state, object_pipeline):
        """Test error handling throughout the pipeline."""
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]

        # Test with invalid location ID
        invalid_location_id = uuid4()

        # Should handle gracefully
        context = await context_collector.collect_generation_context(
            invalid_location_id, "populate_location"
        )

        # Should return fallback context
        assert context is not None
        assert context.location.name == "Unknown Location"

        # Generation should still work with fallback context
        generated_object = await object_generator.generate_object(context)
        assert generated_object is not None

    @pytest.mark.asyncio
    async def test_object_archetype_integration(self, world_state, object_pipeline):
        """Test integration with object archetypes."""
        theme_manager = object_pipeline["theme_manager"]
        context_collector = object_pipeline["context_collector"]
        object_generator = object_pipeline["object_generator"]

        village_id = list(world_state.locations.keys())[0]

        # Get available object types for village
        available_types = await theme_manager.get_available_object_types("Village")
        assert len(available_types) > 0

        # Generate context and object
        context = await context_collector.collect_generation_context(
            village_id, "populate_location"
        )
        generated_object = await object_generator.generate_object(context)

        # Object type should be one of the available types
        assert (
            generated_object.properties.object_type in available_types
            or generated_object.properties.object_type
            in ["container", "tool", "weapon"]
        )  # Fallbacks

    @pytest.mark.asyncio
    async def test_cultural_variations_integration(self, world_state, object_pipeline):
        """Test cultural variations integration."""
        theme_manager = object_pipeline["theme_manager"]

        village_id = list(world_state.locations.keys())[0]
        forest_id = list(world_state.locations.keys())[1]

        village = world_state.locations[village_id]
        forest = world_state.locations[forest_id]

        # Create base properties
        from game_loop.core.models.object_models import ObjectProperties

        base_properties = ObjectProperties(
            name="iron sword", object_type="weapon", material="unknown", value=100
        )

        # Apply cultural variations for different themes
        village_properties = await theme_manager.generate_cultural_variations(
            base_properties, village
        )
        forest_properties = await theme_manager.generate_cultural_variations(
            base_properties, forest
        )

        # Should have different characteristics
        assert village_properties.material != "unknown"
        assert forest_properties.material != "unknown"

        # Values should be adjusted for themes
        assert village_properties.value != forest_properties.value
