"""
Integration tests for the complete location generation pipeline.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio

from game_loop.core.models.location_models import (
    GeneratedLocation,
    LocationGenerationContext,
    LocationTheme,
)
from game_loop.core.models.navigation_models import ExpansionPoint
from game_loop.core.world.context_collector import LocationContextCollector
from game_loop.core.world.location_generator import LocationGenerator
from game_loop.core.world.location_storage import LocationStorage
from game_loop.core.world.theme_manager import LocationThemeManager
from game_loop.state.models import Location, NonPlayerCharacter, WorldObject, WorldState


@pytest.fixture
def world_state_with_sample_data():
    """Create a world state with sample location data."""
    world_state = WorldState()

    # Create a starting location
    signpost_id = uuid4()
    guard_id = uuid4()

    signpost = WorldObject(
        object_id=signpost_id,
        name="signpost",
        description="A wooden signpost pointing deeper into the forest",
    )

    guard = NonPlayerCharacter(
        npc_id=guard_id,
        name="forest guard",
        description="A friendly forest guard watching the entrance",
    )

    starting_location = Location(
        location_id=uuid4(),
        name="Forest Entrance",
        description="A peaceful entrance to the deep forest.",
        connections={"north": uuid4()},
        objects={signpost_id: signpost},
        npcs={guard_id: guard},
        state_flags={"visit_count": 3, "theme": "Forest"},
    )

    # Create a connected location
    tree_id = uuid4()

    ancient_tree = WorldObject(
        object_id=tree_id,
        name="ancient tree",
        description="An ancient oak tree with gnarled branches",
    )

    connected_location = Location(
        location_id=list(starting_location.connections.values())[0],
        name="Deep Grove",
        description="A grove deep in the forest.",
        connections={"south": starting_location.location_id},
        objects={tree_id: ancient_tree},
        npcs={},
        state_flags={"theme": "Forest"},
    )

    world_state.locations[starting_location.location_id] = starting_location
    world_state.locations[connected_location.location_id] = connected_location

    return world_state, starting_location, connected_location


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for the integration test."""
    session_factory = AsyncMock()
    embedding_manager = AsyncMock()
    ollama_client = Mock()

    return {
        "session_factory": session_factory,
        "embedding_manager": embedding_manager,
        "ollama_client": ollama_client,
    }


@pytest_asyncio.fixture
async def integration_system(world_state_with_sample_data, mock_dependencies):
    """Create the complete location generation system for integration testing."""
    world_state, starting_location, connected_location = world_state_with_sample_data

    # Create components
    theme_manager = LocationThemeManager(
        world_state, mock_dependencies["session_factory"]
    )

    context_collector = LocationContextCollector(
        world_state, mock_dependencies["session_factory"]
    )

    location_storage = LocationStorage(
        mock_dependencies["session_factory"], mock_dependencies["embedding_manager"]
    )

    location_generator = LocationGenerator(
        ollama_client=mock_dependencies["ollama_client"],
        world_state=world_state,
        theme_manager=theme_manager,
        context_collector=context_collector,
        location_storage=location_storage,
    )

    return {
        "generator": location_generator,
        "theme_manager": theme_manager,
        "context_collector": context_collector,
        "location_storage": location_storage,
        "world_state": world_state,
        "starting_location": starting_location,
        "connected_location": connected_location,
    }


class TestLocationGenerationIntegration:
    """Integration tests for the complete location generation pipeline."""

    @pytest.mark.asyncio
    async def test_full_generation_pipeline(
        self, integration_system, mock_dependencies
    ):
        """Test the complete location generation pipeline from expansion point to stored location."""
        system = integration_system

        # Create expansion point
        expansion_point = ExpansionPoint(
            location_id=system["starting_location"].location_id,
            direction="west",
            priority=0.8,
            context={"location_name": system["starting_location"].name},
        )

        # Mock theme loading
        forest_theme = LocationTheme(
            name="Forest",
            description="Dense woodland area",
            visual_elements=["trees", "leaves", "shadows"],
            atmosphere="peaceful",
            typical_objects=["log", "mushroom", "rock"],
            typical_npcs=["rabbit", "hermit"],
            generation_parameters={"complexity": "medium"},
            theme_id=uuid4(),
        )

        # Mock theme manager methods
        system["theme_manager"].get_all_themes = AsyncMock(return_value=[forest_theme])
        system["theme_manager"].determine_location_theme = AsyncMock(
            return_value=forest_theme
        )
        system["theme_manager"].validate_theme_consistency = AsyncMock(
            return_value=True
        )

        # Mock context collector methods
        system["context_collector"]._get_world_themes = AsyncMock(
            return_value=[forest_theme]
        )

        # Mock storage methods
        system["location_storage"].get_cached_generation = AsyncMock(return_value=None)
        system["location_storage"].generate_context_hash = Mock(
            return_value="test_hash"
        )
        system["location_storage"].cache_generation_result = AsyncMock()

        # Mock successful storage
        from game_loop.core.models.location_models import StorageResult

        storage_result = StorageResult(
            success=True,
            location_id=uuid4(),
            storage_time_ms=500,
            embedding_generated=True,
        )
        system["location_storage"].store_generated_location = AsyncMock(
            return_value=storage_result
        )

        # Mock LLM response
        import json

        mock_llm_response = {
            "response": json.dumps(
                {
                    "name": "Western Clearing",
                    "description": "A peaceful clearing to the west of the forest entrance, surrounded by tall trees and filled with dappled sunlight.",
                    "short_description": "A western clearing",
                    "location_type": "clearing",
                    "atmosphere": "peaceful",
                    "objects": ["fallen log", "wildflowers"],
                    "potential_npcs": ["forest sprite"],
                    "connections": {"south": "hidden_path"},
                    "special_features": ["sunbeam patterns"],
                }
            )
        }
        mock_dependencies["ollama_client"].generate.return_value = mock_llm_response

        # Execute the full pipeline

        # 1. Collect context
        context = await system["context_collector"].collect_expansion_context(
            expansion_point
        )

        # 2. Generate location
        generated_location = await system["generator"].generate_location(context)

        # 3. Store location
        storage_result = await system["location_storage"].store_generated_location(
            generated_location
        )

        # Verify the results
        assert isinstance(context, LocationGenerationContext)
        assert context.expansion_point == expansion_point
        assert len(context.adjacent_locations) > 0

        assert isinstance(generated_location, GeneratedLocation)
        assert generated_location.name == "Western Clearing"
        assert generated_location.theme.name == "Forest"
        assert "fallen log" in generated_location.objects
        assert "forest sprite" in generated_location.npcs

        assert storage_result.success is True
        assert storage_result.location_id is not None

        # Verify component interactions
        system["theme_manager"].determine_location_theme.assert_called_once()
        system["context_collector"]._get_world_themes.assert_called_once()
        mock_dependencies["ollama_client"].generate.assert_called_once()
        system["location_storage"].store_generated_location.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_collection_with_adjacent_locations(self, integration_system):
        """Test that context collection properly gathers information from adjacent locations."""
        system = integration_system

        expansion_point = ExpansionPoint(
            location_id=system["starting_location"].location_id,
            direction="east",  # Different direction from existing connection
            priority=0.7,
            context={"location_name": system["starting_location"].name},
        )

        # Mock theme loading
        forest_theme = LocationTheme(
            name="Forest",
            description="Dense woodland",
            visual_elements=["trees"],
            atmosphere="peaceful",
            typical_objects=["log"],
            typical_npcs=["hermit"],
            generation_parameters={},
            theme_id=uuid4(),
        )

        system["context_collector"]._get_world_themes = AsyncMock(
            return_value=[forest_theme]
        )

        context = await system["context_collector"].collect_expansion_context(
            expansion_point
        )

        # Should have gathered adjacent location information
        assert len(context.adjacent_locations) > 0

        # Should include the connected location
        adjacent_names = [adj.name for adj in context.adjacent_locations]
        assert system["connected_location"].name in adjacent_names

        # Should have identified the theme
        adjacent_themes = [adj.theme for adj in context.adjacent_locations]
        assert "Forest" in adjacent_themes

    @pytest.mark.asyncio
    async def test_theme_consistency_validation(self, integration_system):
        """Test theme consistency validation across the pipeline."""
        system = integration_system

        # Create themes with different compatibility
        forest_theme = LocationTheme(
            name="Forest",
            description="Forest area",
            visual_elements=["trees"],
            atmosphere="peaceful",
            typical_objects=["log"],
            typical_npcs=["hermit"],
            generation_parameters={},
            theme_id=uuid4(),
        )

        mountain_theme = LocationTheme(
            name="Mountain",
            description="Rocky area",
            visual_elements=["rocks"],
            atmosphere="rugged",
            typical_objects=["boulder"],
            typical_npcs=["climber"],
            generation_parameters={},
            theme_id=uuid4(),
        )

        # Test compatible theme validation
        result = await system["theme_manager"].validate_theme_consistency(
            forest_theme, [forest_theme]  # Same theme should be compatible
        )
        assert result is True

        # Mock transition rules for forest->mountain
        from game_loop.core.models.location_models import ThemeTransitionRules

        transition_rules = ThemeTransitionRules(
            from_theme="Forest",
            to_theme="Mountain",
            compatibility_score=0.6,  # Moderate compatibility
            transition_requirements=["elevation_change"],
        )

        system["theme_manager"].get_theme_transition_rules = AsyncMock(
            return_value=transition_rules
        )

        result = await system["theme_manager"].validate_theme_consistency(
            mountain_theme, [forest_theme]
        )
        assert result is True  # 0.6 > 0.3 threshold

    @pytest.mark.asyncio
    async def test_caching_integration(self, integration_system, mock_dependencies):
        """Test that caching works across the generation pipeline."""
        system = integration_system

        expansion_point = ExpansionPoint(
            location_id=system["starting_location"].location_id,
            direction="west",
            priority=0.8,
            context={"location_name": system["starting_location"].name},
        )

        # Create cached location
        forest_theme = LocationTheme(
            name="Forest",
            description="Forest area",
            visual_elements=["trees"],
            atmosphere="peaceful",
            typical_objects=["log"],
            typical_npcs=["hermit"],
            generation_parameters={},
            theme_id=uuid4(),
        )

        cached_location = GeneratedLocation(
            name="Cached Grove",
            description="A cached forest grove",
            theme=forest_theme,
            location_type="clearing",
            objects=["cached_object"],
            npcs=["cached_npc"],
            connections={},
            metadata={},
        )

        # Mock cache hit
        from datetime import datetime, timedelta

        from game_loop.core.models.location_models import CachedGeneration

        cached_generation = CachedGeneration(
            context_hash="test_hash",
            generated_location=cached_location,
            cache_expires_at=datetime.now() + timedelta(hours=1),
            usage_count=0,
        )

        system["location_storage"].get_cached_generation = AsyncMock(
            return_value=cached_generation
        )
        system["location_storage"].generate_context_hash = Mock(
            return_value="test_hash"
        )

        # Mock theme manager
        system["theme_manager"].get_all_themes = AsyncMock(return_value=[forest_theme])
        system["context_collector"]._get_world_themes = AsyncMock(
            return_value=[forest_theme]
        )

        # Collect context
        context = await system["context_collector"].collect_expansion_context(
            expansion_point
        )

        # Generate location (should use cache)
        result = await system["generator"].generate_location(context)

        # Should return cached location
        assert result == cached_location

        # LLM should not be called
        mock_dependencies["ollama_client"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(
        self, integration_system, mock_dependencies
    ):
        """Test error handling and fallback mechanisms in the pipeline."""
        system = integration_system

        expansion_point = ExpansionPoint(
            location_id=system["starting_location"].location_id,
            direction="west",
            priority=0.8,
            context={"location_name": system["starting_location"].name},
        )

        forest_theme = LocationTheme(
            name="Forest",
            description="Forest area",
            visual_elements=["trees"],
            atmosphere="peaceful",
            typical_objects=["log"],
            typical_npcs=["hermit"],
            generation_parameters={},
            theme_id=uuid4(),
        )

        # Mock theme and storage setup
        system["theme_manager"].get_all_themes = AsyncMock(return_value=[forest_theme])
        system["theme_manager"].determine_location_theme = AsyncMock(
            return_value=forest_theme
        )
        system["theme_manager"].validate_theme_consistency = AsyncMock(
            return_value=True
        )
        system["context_collector"]._get_world_themes = AsyncMock(
            return_value=[forest_theme]
        )
        system["location_storage"].get_cached_generation = AsyncMock(return_value=None)
        system["location_storage"].generate_context_hash = Mock(
            return_value="test_hash"
        )
        system["location_storage"].cache_generation_result = AsyncMock()

        # Mock LLM failure
        mock_dependencies["ollama_client"].generate.side_effect = Exception(
            "LLM Connection Error"
        )

        # Execute pipeline
        context = await system["context_collector"].collect_expansion_context(
            expansion_point
        )

        # Should still generate a location using fallback
        result = await system["generator"].generate_location(context)

        # Should have fallback location
        assert isinstance(result, GeneratedLocation)
        assert result.theme.name == "Forest"
        assert "Forest" in result.name  # Fallback uses theme name

    @pytest.mark.asyncio
    async def test_database_integration_simulation(
        self, integration_system, mock_dependencies
    ):
        """Test database integration with mocked database responses."""
        system = integration_system
        session_factory, session = mock_dependencies["session_factory"], AsyncMock()
        mock_dependencies["session_factory"].return_value.__aenter__.return_value = (
            session
        )

        # Mock theme loading from database
        forest_theme = LocationTheme(
            name="Forest",
            description="Dense woodland area",
            visual_elements=["trees", "leaves"],
            atmosphere="peaceful",
            typical_objects=["log", "mushroom"],
            typical_npcs=["rabbit", "hermit"],
            generation_parameters={"complexity": "medium"},
            theme_id=uuid4(),
        )

        # Mock database theme query response
        mock_theme_row = Mock()
        mock_theme_row.name = forest_theme.name
        mock_theme_row.description = forest_theme.description
        mock_theme_row.visual_elements = forest_theme.visual_elements
        mock_theme_row.atmosphere = forest_theme.atmosphere
        mock_theme_row.typical_objects = forest_theme.typical_objects
        mock_theme_row.typical_npcs = forest_theme.typical_npcs
        mock_theme_row.generation_parameters = forest_theme.generation_parameters
        mock_theme_row.theme_id = forest_theme.theme_id
        mock_theme_row.parent_theme_id = None
        mock_theme_row.created_at = None

        mock_theme_result = Mock()
        mock_theme_result.__iter__ = lambda self: iter([mock_theme_row])
        session.execute.return_value = mock_theme_result
        session.commit.return_value = None

        # Load themes from "database"
        themes = await system["theme_manager"].get_all_themes()

        assert len(themes) == 1
        assert themes[0].name == "Forest"
        assert themes[0].theme_id == forest_theme.theme_id

        # Verify database interaction
        session.execute.assert_called()


class TestThemeSystemIntegration:
    """Integration tests for the theme system."""

    @pytest.mark.asyncio
    async def test_theme_inheritance_and_transitions(self, mock_dependencies):
        """Test theme inheritance and transition validation."""
        world_state = WorldState()
        theme_manager = LocationThemeManager(
            world_state, mock_dependencies["session_factory"]
        )

        # Create parent and child themes
        parent_theme = LocationTheme(
            name="Nature",
            description="Natural environments",
            visual_elements=["organic", "natural"],
            atmosphere="serene",
            typical_objects=["natural_object"],
            typical_npcs=["nature_dweller"],
            generation_parameters={"base_complexity": "medium"},
            theme_id=uuid4(),
        )

        child_theme = LocationTheme(
            name="Forest",
            description="Forest environments",
            visual_elements=["trees", "leaves"],
            atmosphere="peaceful",
            typical_objects=["log", "mushroom"],
            typical_npcs=["forest_animal"],
            generation_parameters={"complexity": "medium"},
            theme_id=uuid4(),
            parent_theme_id=parent_theme.theme_id,
        )

        # Test theme-specific content generation
        content = await theme_manager.generate_theme_specific_content(
            child_theme, "clearing"
        )

        assert content.theme_name == "Forest"
        assert "log" in content.objects
        assert "mushroom" in content.objects
        assert "forest_animal" in content.npcs

        # Test transition rules
        transition_rules = await theme_manager.get_theme_transition_rules(
            "Forest", "Village"
        )

        assert isinstance(transition_rules, ThemeTransitionRules)
        assert transition_rules.from_theme == "Forest"
        assert transition_rules.to_theme == "Village"
        assert 0.0 <= transition_rules.compatibility_score <= 1.0


@pytest.mark.asyncio
async def test_performance_metrics_collection(integration_system, mock_dependencies):
    """Test that performance metrics are collected during generation."""
    system = integration_system

    expansion_point = ExpansionPoint(
        location_id=system["starting_location"].location_id,
        direction="west",
        priority=0.8,
        context={"location_name": system["starting_location"].name},
    )

    forest_theme = LocationTheme(
        name="Forest",
        description="Forest area",
        visual_elements=["trees"],
        atmosphere="peaceful",
        typical_objects=["log"],
        typical_npcs=["hermit"],
        generation_parameters={},
        theme_id=uuid4(),
    )

    # Mock all dependencies
    system["theme_manager"].get_all_themes = AsyncMock(return_value=[forest_theme])
    system["theme_manager"].determine_location_theme = AsyncMock(
        return_value=forest_theme
    )
    system["theme_manager"].validate_theme_consistency = AsyncMock(return_value=True)
    system["context_collector"]._get_world_themes = AsyncMock(
        return_value=[forest_theme]
    )
    system["location_storage"].get_cached_generation = AsyncMock(return_value=None)
    system["location_storage"].generate_context_hash = Mock(return_value="test_hash")
    system["location_storage"].cache_generation_result = AsyncMock()

    import json

    mock_llm_response = {
        "response": json.dumps(
            {
                "name": "Test Location",
                "description": "A test location",
                "location_type": "clearing",
                "atmosphere": "peaceful",
                "objects": ["test_object"],
                "potential_npcs": ["test_npc"],
                "connections": {},
                "special_features": [],
            }
        )
    }
    mock_dependencies["ollama_client"].generate.return_value = mock_llm_response

    # Generate location
    context = await system["context_collector"].collect_expansion_context(
        expansion_point
    )
    result = await system["generator"].generate_location(context)

    # Check metrics collection
    metrics = system["generator"].get_generation_metrics()

    assert len(metrics) > 0
    last_metric = metrics[-1]
    assert last_metric.generation_time_ms > 0
    assert last_metric.total_time_ms > 0
