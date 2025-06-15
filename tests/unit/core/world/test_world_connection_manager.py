"""
Unit tests for World Connection Manager.

Tests the main connection generation orchestrator and its integration with other components.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from game_loop.core.models.connection_models import (
    ConnectionProperties,
    ConnectionValidationResult,
    GeneratedConnection,
)
from game_loop.core.world.world_connection_manager import WorldConnectionManager
from game_loop.state.models import Location, WorldState


class TestWorldConnectionManager:
    """Test WorldConnectionManager functionality."""

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
        mock_client = AsyncMock()
        # Mock response for connection description generation
        mock_client.generate_completion.return_value = {
            "response": json.dumps({
                "description": "A well-crafted stone bridge spans gracefully between the forest and village, with moss-covered railings that speak of age and endurance.",
                "travel_time": 60,
                "difficulty": 3,
                "requirements": [],
                "special_features": ["scenic_view", "weathered_stone"],
                "atmosphere": "peaceful transition",
                "visibility_notes": "Clearly visible from both sides",
            })
        }
        return mock_client

    @pytest.fixture
    def template_env(self):
        """Create mock template environment."""
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "Mock template prompt"
        mock_env.get_template.return_value = mock_template
        return mock_env

    @pytest.fixture
    def connection_manager(
        self, world_state, session_factory, llm_client, template_env
    ):
        """Create WorldConnectionManager instance."""
        return WorldConnectionManager(
            world_state=world_state,
            session_factory=session_factory,
            llm_client=llm_client,
            template_env=template_env,
        )

    @pytest.fixture
    def forest_location(self):
        """Create forest location for testing."""
        return Location(
            location_id=uuid4(),
            name="Whispering Woods",
            description="An ancient forest with towering trees",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Forest"},
        )

    @pytest.fixture
    def village_location(self):
        """Create village location for testing."""
        return Location(
            location_id=uuid4(),
            name="Millbrook Village",
            description="A peaceful farming village",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Village"},
        )

    @pytest.fixture
    def setup_world_state(self, world_state, forest_location, village_location):
        """Setup world state with test locations."""
        world_state.locations[forest_location.location_id] = forest_location
        world_state.locations[village_location.location_id] = village_location
        return world_state

    @pytest.mark.asyncio
    async def test_generate_connection_basic(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test basic connection generation."""
        connection = await connection_manager.generate_connection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            purpose="expand_world",
        )

        assert isinstance(connection, GeneratedConnection)
        assert connection.source_location_id == forest_location.location_id
        assert connection.target_location_id == village_location.location_id
        assert isinstance(connection.properties, ConnectionProperties)
        assert connection.properties.connection_type in [
            "passage",
            "bridge",
            "path",
            "road",
            "tunnel",
            "portal",
        ]
        assert 1 <= connection.properties.difficulty <= 10
        assert connection.properties.travel_time > 0
        assert len(connection.properties.description) > 0

    @pytest.mark.asyncio
    async def test_generate_connection_with_metadata(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test that generated connection includes proper metadata."""
        connection = await connection_manager.generate_connection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            purpose="quest_path",
        )

        assert "generation_purpose" in connection.metadata
        assert connection.metadata["generation_purpose"] == "quest_path"
        assert "connection_type" in connection.metadata
        assert "generation_context" in connection.metadata
        assert "validation_result" in connection.metadata

    @pytest.mark.asyncio
    async def test_generate_connection_caching(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test that connection generation uses caching."""
        # Generate connection twice
        connection1 = await connection_manager.generate_connection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            purpose="expand_world",
        )

        connection2 = await connection_manager.generate_connection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            purpose="expand_world",
        )

        # Should return the same cached connection
        assert connection1.connection_id == connection2.connection_id
        assert connection1.properties.description == connection2.properties.description

    @pytest.mark.asyncio
    async def test_create_connection_properties(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test connection properties creation."""
        # First need to collect context
        context = await connection_manager.context_collector.collect_generation_context(
            forest_location.location_id, village_location.location_id, "expand_world"
        )

        properties = await connection_manager.create_connection_properties(
            "bridge", context
        )

        assert isinstance(properties, ConnectionProperties)
        assert properties.connection_type == "bridge"
        assert 1 <= properties.difficulty <= 10
        assert properties.travel_time > 0
        assert len(properties.description) > 0
        assert properties.visibility in [
            "visible",
            "hidden",
            "secret",
            "partially_hidden",
        ]

    @pytest.mark.asyncio
    async def test_validate_connection_valid(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test validation of a valid connection."""
        # Create a valid connection
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=60,
            description="A well-built stone bridge",
            visibility="visible",
            requirements=[],
        )

        connection = GeneratedConnection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            properties=properties,
        )

        context = await connection_manager.context_collector.collect_generation_context(
            forest_location.location_id, village_location.location_id, "expand_world"
        )

        result = await connection_manager.validate_connection(connection, context)

        assert isinstance(result, ConnectionValidationResult)
        assert result.is_valid is True
        assert len(result.validation_errors) == 0
        assert 0.0 <= result.consistency_score <= 1.0
        assert 0.0 <= result.logical_soundness <= 1.0
        assert 0.0 <= result.terrain_compatibility <= 1.0

    @pytest.mark.asyncio
    async def test_validate_connection_invalid(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test validation of an invalid connection."""
        # Create a connection and then make it invalid
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,  # Valid initially
            travel_time=60,
            description="Valid description",  # Valid initially
            visibility="visible",
            requirements=[],
        )

        # Manually set invalid values after creation to bypass validation
        properties.description = ""  # Make it invalid

        connection = GeneratedConnection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            properties=properties,
        )

        context = await connection_manager.context_collector.collect_generation_context(
            forest_location.location_id, village_location.location_id, "expand_world"
        )

        result = await connection_manager.validate_connection(connection, context)

        assert isinstance(result, ConnectionValidationResult)
        assert result.is_valid is False
        assert len(result.validation_errors) > 0
        assert "empty" in " ".join(result.validation_errors).lower()

    @pytest.mark.asyncio
    async def test_update_world_graph(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test updating world connectivity graph."""
        properties = ConnectionProperties(
            connection_type="path",
            difficulty=2,
            travel_time=90,
            description="A forest path",
            visibility="visible",
            requirements=[],
        )

        connection = GeneratedConnection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            properties=properties,
        )

        success = await connection_manager.update_world_graph(connection)

        assert success is True

        # Check that graph was updated
        graph = connection_manager.connectivity_graph
        assert forest_location.location_id in graph.nodes
        assert village_location.location_id in graph.nodes

        # Check that connection exists in graph
        connections_from_forest = graph.get_connections_from(
            forest_location.location_id
        )
        assert len(connections_from_forest) > 0
        assert any(
            conn.connection_id == connection.connection_id
            for conn in connections_from_forest
        )

    @pytest.mark.asyncio
    async def test_find_connection_opportunities(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test finding connection opportunities."""
        opportunities = await connection_manager.find_connection_opportunities(
            forest_location.location_id
        )

        assert isinstance(opportunities, list)
        # Should find the village as a potential target
        target_ids = [target_id for target_id, score in opportunities]
        assert village_location.location_id in target_ids

        # Check that scores are reasonable
        for target_id, score in opportunities:
            assert isinstance(target_id, type(village_location.location_id))
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_generate_connection_description_with_llm(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test LLM-based connection description generation."""
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=60,
            description="Initial description",
            visibility="visible",
            requirements=[],
        )

        context = await connection_manager.context_collector.collect_generation_context(
            forest_location.location_id, village_location.location_id, "expand_world"
        )

        description = await connection_manager.generate_connection_description(
            properties, context
        )

        assert isinstance(description, str)
        assert len(description) > 0
        # Should contain LLM-generated content
        assert "bridge" in description.lower()

    @pytest.mark.asyncio
    async def test_generate_connection_description_fallback(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test fallback when LLM description generation fails."""
        # Mock LLM to raise an exception
        connection_manager.llm_client.generate_response.side_effect = Exception(
            "LLM error"
        )

        properties = ConnectionProperties(
            connection_type="path",
            difficulty=2,
            travel_time=90,
            description="Initial description",
            visibility="visible",
            requirements=[],
        )

        context = await connection_manager.context_collector.collect_generation_context(
            forest_location.location_id, village_location.location_id, "expand_world"
        )

        description = await connection_manager.generate_connection_description(
            properties, context
        )

        assert isinstance(description, str)
        assert len(description) > 0
        # Should fallback to theme manager description

    @pytest.mark.asyncio
    async def test_generate_connection_error_handling(
        self, connection_manager, world_state
    ):
        """Test error handling when locations don't exist."""
        nonexistent_id1 = uuid4()
        nonexistent_id2 = uuid4()

        # Should not raise exception, should return fallback connection
        connection = await connection_manager.generate_connection(
            source_location_id=nonexistent_id1,
            target_location_id=nonexistent_id2,
            purpose="expand_world",
        )

        assert isinstance(connection, GeneratedConnection)
        assert connection.source_location_id == nonexistent_id1
        assert connection.target_location_id == nonexistent_id2
        # System should handle missing locations gracefully by creating a fallback context
        # The connection should still be valid
        assert connection.properties.connection_type in ["passage", "bridge", "path", "road", "tunnel", "portal"]

    @pytest.mark.asyncio
    async def test_connection_enhancement_based_on_context(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test that connections are enhanced based on generation context."""
        # Test quest path purpose (should be more difficult/interesting)
        quest_connection = await connection_manager.generate_connection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            purpose="quest_path",
        )

        # Test exploration purpose
        exploration_connection = await connection_manager.generate_connection(
            source_location_id=village_location.location_id,  # Reverse direction to avoid cache
            target_location_id=forest_location.location_id,
            purpose="exploration",
        )

        assert isinstance(quest_connection, GeneratedConnection)
        assert isinstance(exploration_connection, GeneratedConnection)

        # Both should have valid properties
        assert quest_connection.properties.connection_type in [
            "passage",
            "bridge",
            "path",
            "road",
            "tunnel",
            "portal",
        ]
        assert exploration_connection.properties.connection_type in [
            "passage",
            "bridge",
            "path",
            "road",
            "tunnel",
            "portal",
        ]

    @pytest.mark.asyncio
    async def test_connection_validation_integration(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test that generated connections are automatically validated."""
        connection = await connection_manager.generate_connection(
            source_location_id=forest_location.location_id,
            target_location_id=village_location.location_id,
            purpose="expand_world",
        )

        # Check that validation metadata exists
        assert "validation_result" in connection.metadata
        validation_metadata = connection.metadata["validation_result"]

        assert "is_valid" in validation_metadata
        assert "consistency_score" in validation_metadata
        assert isinstance(validation_metadata["is_valid"], bool)
        assert isinstance(validation_metadata["consistency_score"], (int, float))

    @pytest.mark.asyncio
    async def test_different_connection_purposes(
        self, connection_manager, setup_world_state, forest_location, village_location
    ):
        """Test that different purposes generate appropriate connections."""
        purposes = [
            "expand_world",
            "quest_path",
            "exploration",
            "narrative_enhancement",
        ]

        for purpose in purposes:
            # Use different source/target combinations to avoid caching
            if purpose == "expand_world":
                source, target = forest_location, village_location
            elif purpose == "quest_path":
                source, target = village_location, forest_location
            elif purpose == "exploration":
                # Create a new location for variety
                new_location = Location(
                    location_id=uuid4(),
                    name="Test Location",
                    description="Test",
                    connections={},
                    objects={},
                    npcs={},
                    state_flags={"theme": "Mountain"},
                )
                connection_manager.world_state.locations[new_location.location_id] = (
                    new_location
                )
                source, target = forest_location, new_location
            else:  # narrative_enhancement
                new_location2 = Location(
                    location_id=uuid4(),
                    name="Test Location 2",
                    description="Test",
                    connections={},
                    objects={},
                    npcs={},
                    state_flags={"theme": "City"},
                )
                connection_manager.world_state.locations[new_location2.location_id] = (
                    new_location2
                )
                source, target = village_location, new_location2

            connection = await connection_manager.generate_connection(
                source_location_id=source.location_id,
                target_location_id=target.location_id,
                purpose=purpose,
            )

            assert isinstance(connection, GeneratedConnection)
            assert connection.metadata["generation_purpose"] == purpose
