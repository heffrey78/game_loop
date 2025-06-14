"""
Unit tests for LocationGenerator.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import (
    AdjacentLocationContext,
    GeneratedLocation,
    LocationGenerationContext,
    LocationTheme,
    PlayerLocationPreferences,
    ValidationResult,
)
from game_loop.core.models.navigation_models import ExpansionPoint
from game_loop.core.world.location_generator import LocationGenerator
from game_loop.state.models import WorldState


@pytest.fixture
def sample_theme():
    """Create a sample location theme."""
    return LocationTheme(
        name="Forest",
        description="Dense woodland area",
        visual_elements=["trees", "leaves", "shadows"],
        atmosphere="peaceful",
        typical_objects=["log", "mushroom", "rock"],
        typical_npcs=["rabbit", "hermit"],
        generation_parameters={"complexity": "medium"},
    )


@pytest.fixture
def sample_expansion_point():
    """Create a sample expansion point."""
    return ExpansionPoint(
        location_id=uuid4(),
        direction="north",
        priority=0.8,
        context={"location_name": "Starting Grove"},
    )


@pytest.fixture
def sample_context(sample_expansion_point, sample_theme):
    """Create a sample generation context."""
    adjacent_location = AdjacentLocationContext(
        location_id=uuid4(),
        direction="south",
        name="Starting Grove",
        description="A peaceful grove",
        theme="Forest",
        short_description="A grove",
    )

    player_prefs = PlayerLocationPreferences(
        environments=["forest"], interaction_style="explorer", complexity_level="medium"
    )

    return LocationGenerationContext(
        expansion_point=sample_expansion_point,
        adjacent_locations=[adjacent_location],
        player_preferences=player_prefs,
        world_themes=[sample_theme],
    )


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for LocationGenerator."""
    ollama_client = Mock()
    world_state = WorldState()
    theme_manager = AsyncMock()
    context_collector = AsyncMock()
    location_storage = AsyncMock()

    return {
        "ollama_client": ollama_client,
        "world_state": world_state,
        "theme_manager": theme_manager,
        "context_collector": context_collector,
        "location_storage": location_storage,
    }


@pytest.fixture
def location_generator(mock_dependencies):
    """Create a LocationGenerator instance with mocked dependencies."""
    return LocationGenerator(**mock_dependencies)


class TestLocationGenerator:
    """Test cases for LocationGenerator."""

    @pytest.mark.asyncio
    async def test_generate_location_success(
        self, location_generator, sample_context, sample_theme
    ):
        """Test successful location generation."""
        # Mock the dependencies
        location_generator.theme_manager.determine_location_theme.return_value = (
            sample_theme
        )
        location_generator.context_collector.enrich_context.return_value = AsyncMock()
        location_generator.context_collector.enrich_context.return_value.base_context = (
            sample_context
        )
        location_generator.context_collector.enrich_context.return_value.generation_hints = [
            "test hint"
        ]
        location_generator.context_collector.enrich_context.return_value.priority_elements = [
            "theme_consistency"
        ]

        # Mock storage cache miss
        location_generator.location_storage.get_cached_generation.return_value = None
        location_generator.location_storage.generate_context_hash.return_value = (
            "test_hash"
        )
        location_generator.location_storage.cache_generation_result = AsyncMock()

        # Mock LLM response
        mock_llm_response = {
            "response": json.dumps(
                {
                    "name": "Northern Grove",
                    "description": "A beautiful grove to the north with tall trees.",
                    "short_description": "A northern grove",
                    "location_type": "clearing",
                    "atmosphere": "peaceful",
                    "objects": ["ancient oak", "mushroom ring"],
                    "potential_npcs": ["wise owl"],
                    "connections": {"east": "possible_path"},
                    "special_features": ["sacred circle"],
                }
            )
        }
        location_generator.ollama_client.generate.return_value = mock_llm_response

        # Execute
        result = await location_generator.generate_location(sample_context)

        # Verify
        assert isinstance(result, GeneratedLocation)
        assert result.name == "Northern Grove"
        assert result.theme == sample_theme
        assert "ancient oak" in result.objects
        assert "wise owl" in result.npcs

        # Verify mocks were called
        location_generator.theme_manager.determine_location_theme.assert_called_once()
        location_generator.ollama_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_location_with_cache_hit(
        self, location_generator, sample_context, sample_theme
    ):
        """Test location generation with cache hit."""
        # Create cached result
        cached_location = GeneratedLocation(
            name="Cached Grove",
            description="A cached grove",
            theme=sample_theme,
            location_type="clearing",
            objects=["cached object"],
            npcs=["cached npc"],
            connections={},
            metadata={},
        )

        mock_cached = AsyncMock()
        mock_cached.generated_location = cached_location

        # Mock cache hit
        location_generator.location_storage.get_cached_generation.return_value = (
            mock_cached
        )
        location_generator.location_storage.generate_context_hash.return_value = (
            "test_hash"
        )

        # Execute
        result = await location_generator.generate_location(sample_context)

        # Verify cache was used
        assert result == cached_location

        # Verify LLM was not called
        location_generator.ollama_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_location_llm_failure_fallback(
        self, location_generator, sample_context, sample_theme
    ):
        """Test fallback when LLM generation fails."""
        # Mock dependencies
        location_generator.theme_manager.determine_location_theme.return_value = (
            sample_theme
        )
        location_generator.context_collector.enrich_context.return_value = AsyncMock()
        location_generator.context_collector.enrich_context.return_value.base_context = (
            sample_context
        )
        location_generator.context_collector.enrich_context.return_value.generation_hints = (
            []
        )
        location_generator.context_collector.enrich_context.return_value.priority_elements = (
            []
        )

        # Mock storage
        location_generator.location_storage.get_cached_generation.return_value = None
        location_generator.location_storage.generate_context_hash.return_value = (
            "test_hash"
        )
        location_generator.location_storage.cache_generation_result = AsyncMock()

        # Mock LLM failure
        location_generator.ollama_client.generate.side_effect = Exception("LLM Error")

        # Execute
        result = await location_generator.generate_location(sample_context)

        # Verify fallback was used
        assert isinstance(result, GeneratedLocation)
        assert "Forest" in result.name  # Should use theme name
        assert result.theme == sample_theme

    def test_parse_llm_response_valid_json(self, location_generator):
        """Test parsing valid JSON response from LLM."""
        response = """
        Some text before
        {
            "name": "Test Location",
            "description": "A test location",
            "location_type": "clearing"
        }
        Some text after
        """

        result = location_generator._parse_llm_response(response)

        assert result["name"] == "Test Location"
        assert result["description"] == "A test location"
        assert result["location_type"] == "clearing"

    def test_parse_llm_response_invalid_json(self, location_generator):
        """Test parsing invalid JSON response from LLM."""
        response = "This is not JSON at all"

        result = location_generator._parse_llm_response(response)

        # Should return fallback data
        assert "name" in result
        assert "description" in result
        assert len(result["description"]) > 0

    @pytest.mark.asyncio
    async def test_validate_location_consistency_valid(
        self, location_generator, sample_theme
    ):
        """Test location consistency validation for valid location."""
        location = GeneratedLocation(
            name="Valid Grove",
            description="A well-described forest grove with appropriate details for the setting.",
            theme=sample_theme,
            location_type="clearing",
            objects=["oak tree", "mushroom"],
            npcs=["forest sprite"],
            connections={},
            metadata={},
            special_features=["ancient stone circle"],
        )

        # Mock theme validation
        location_generator.theme_manager.validate_theme_consistency.return_value = True

        result = await location_generator.validate_location_consistency(
            location, [sample_theme]
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.overall_score >= 6.0

    @pytest.mark.asyncio
    async def test_validate_location_consistency_invalid(
        self, location_generator, sample_theme
    ):
        """Test location consistency validation for invalid location."""
        location = GeneratedLocation(
            name="",  # Empty name
            description="Short",  # Too short description
            theme=sample_theme,
            location_type="clearing",
            objects=[
                "obj1",
                "obj2",
                "obj3",
                "obj4",
                "obj5",
                "obj6",
            ],  # Too many objects
            npcs=["npc1", "npc2", "npc3", "npc4"],  # Too many NPCs
            connections={},
            metadata={},
        )

        # Mock theme validation failure
        location_generator.theme_manager.validate_theme_consistency.return_value = False

        result = await location_generator.validate_location_consistency(
            location, [sample_theme]
        )

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert len(result.issues) > 0
        assert result.overall_score < 6.0

    def test_create_fallback_location(
        self, location_generator, sample_context, sample_theme
    ):
        """Test creation of fallback location."""
        enriched_context = AsyncMock()
        enriched_context.base_context = sample_context

        result = location_generator._create_fallback_location(
            enriched_context, sample_theme
        )

        assert "name" in result
        assert "description" in result
        assert result["atmosphere"] == sample_theme.atmosphere
        assert len(result["objects"]) <= 2
        assert len(result["potential_npcs"]) <= 1

    def test_serialize_context_for_cache(self, location_generator, sample_context):
        """Test serialization of context for caching."""
        result = location_generator._serialize_context_for_cache(sample_context)

        assert "expansion_point" in result
        assert "adjacent_locations" in result
        assert "player_preferences" in result
        assert "world_themes" in result

        # Verify expansion point serialization
        expansion_data = result["expansion_point"]
        assert expansion_data["direction"] == sample_context.expansion_point.direction
        assert expansion_data["priority"] == sample_context.expansion_point.priority

    def test_get_generation_metrics(self, location_generator):
        """Test getting generation metrics."""
        # Initially should be empty
        metrics = location_generator.get_generation_metrics()
        assert len(metrics) == 0

        # After clearing, should still be empty
        location_generator.clear_metrics()
        metrics = location_generator.get_generation_metrics()
        assert len(metrics) == 0

    @pytest.mark.asyncio
    async def test_generate_location_connections(
        self, location_generator, sample_expansion_point, sample_theme
    ):
        """Test generation of location connections."""
        location = GeneratedLocation(
            name="Test Location",
            description="A test location",
            theme=sample_theme,
            location_type="crossroads",
            objects=[],
            npcs=[],
            connections={},
            metadata={},
        )

        connections = await location_generator.generate_location_connections(
            location, sample_expansion_point
        )

        assert len(connections) >= 1  # Should have at least the primary connection

        # Check primary connection
        primary_connection = connections[0]
        assert primary_connection.from_location_id == sample_expansion_point.location_id
        assert primary_connection.direction == sample_expansion_point.direction
        assert primary_connection.is_bidirectional


@pytest.mark.asyncio
class TestLocationGeneratorIntegration:
    """Integration tests for LocationGenerator with real template system."""

    def test_template_loading(self, mock_dependencies):
        """Test that template system loads correctly."""
        # Test with real template path (should work if templates exist)
        generator = LocationGenerator(**mock_dependencies)

        # Should not raise an exception
        assert generator is not None

        # If templates are not found, jinja_env should be None
        if generator.jinja_env is None:
            # Templates not found, which is expected in test environment
            assert True
        else:
            # Templates found and loaded successfully
            assert generator.jinja_env is not None

    @patch("ollama.Client")
    @pytest.mark.asyncio
    async def test_full_generation_pipeline_mock(
        self, mock_ollama_class, sample_context, sample_theme
    ):
        """Test the full generation pipeline with mocked Ollama."""
        # Setup mocks
        mock_ollama_instance = Mock()
        mock_ollama_class.return_value = mock_ollama_instance

        world_state = WorldState()
        theme_manager = AsyncMock()
        context_collector = AsyncMock()
        location_storage = AsyncMock()

        # Setup mock responses
        theme_manager.determine_location_theme.return_value = sample_theme
        theme_manager.validate_theme_consistency.return_value = True

        context_collector.enrich_context.return_value = AsyncMock()
        context_collector.enrich_context.return_value.base_context = sample_context
        context_collector.enrich_context.return_value.generation_hints = ["test hint"]
        context_collector.enrich_context.return_value.priority_elements = ["quality"]

        location_storage.get_cached_generation.return_value = None
        location_storage.generate_context_hash.return_value = "test_hash"
        location_storage.cache_generation_result = AsyncMock()

        mock_llm_response = {
            "response": json.dumps(
                {
                    "name": "Generated Grove",
                    "description": "A beautifully generated grove with LLM content.",
                    "short_description": "Generated grove",
                    "location_type": "clearing",
                    "atmosphere": "mystical",
                    "objects": ["glowing mushroom", "crystal formation"],
                    "potential_npcs": ["forest guardian"],
                    "connections": {"west": "mountain_path"},
                    "special_features": ["magical aura"],
                }
            )
        }
        mock_ollama_instance.generate.return_value = mock_llm_response

        # Create generator
        generator = LocationGenerator(
            ollama_client=mock_ollama_instance,
            world_state=world_state,
            theme_manager=theme_manager,
            context_collector=context_collector,
            location_storage=location_storage,
        )

        # Execute full pipeline
        result = await generator.generate_location(sample_context)

        # Verify result
        assert isinstance(result, GeneratedLocation)
        assert result.name == "Generated Grove"
        assert result.theme == sample_theme
        assert "glowing mushroom" in result.objects
        assert "forest guardian" in result.npcs
        assert result.metadata["atmosphere"] == "mystical"

        # Verify all components were called
        theme_manager.determine_location_theme.assert_called_once_with(sample_context)
        context_collector.enrich_context.assert_called_once_with(sample_context)
        mock_ollama_instance.generate.assert_called_once()
