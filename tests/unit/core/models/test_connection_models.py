"""
Unit tests for Connection Data Models.

Tests all connection model classes for validation, creation, and behavior.
"""

from datetime import datetime
from uuid import UUID, uuid4

import pytest

from game_loop.core.models.connection_models import (
    ConnectionArchetype,
    ConnectionGenerationContext,
    ConnectionGenerationRequest,
    ConnectionMetrics,
    ConnectionProperties,
    ConnectionSearchCriteria,
    ConnectionStorageResult,
    ConnectionValidationResult,
    GeneratedConnection,
    WorldConnectivityGraph,
)
from game_loop.state.models import Location


class TestConnectionProperties:
    """Test ConnectionProperties data model."""

    def test_connection_properties_creation(self):
        """Test basic connection properties creation."""
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=45,
            description="A stone bridge spans the gap",
            visibility="visible",
            requirements=["steady_footing"],
        )

        assert properties.connection_type == "bridge"
        assert properties.difficulty == 3
        assert properties.travel_time == 45
        assert properties.description == "A stone bridge spans the gap"
        assert properties.visibility == "visible"
        assert properties.requirements == ["steady_footing"]
        assert properties.reversible is True  # default
        assert properties.condition_flags == {}  # default
        assert properties.special_features == []  # default

    def test_connection_properties_validation(self):
        """Test connection properties validation."""
        # Test invalid difficulty
        with pytest.raises(
            ValueError, match="Connection difficulty must be between 1 and 10"
        ):
            ConnectionProperties(
                connection_type="passage",
                difficulty=15,
                travel_time=30,
                description="Test",
                visibility="visible",
                requirements=[],
            )

        # Test negative travel time
        with pytest.raises(ValueError, match="Travel time cannot be negative"):
            ConnectionProperties(
                connection_type="passage",
                difficulty=5,
                travel_time=-10,
                description="Test",
                visibility="visible",
                requirements=[],
            )

        # Test invalid visibility
        with pytest.raises(ValueError, match="Invalid visibility"):
            ConnectionProperties(
                connection_type="passage",
                difficulty=5,
                travel_time=30,
                description="Test",
                visibility="invalid",
                requirements=[],
            )

        # Test invalid connection type
        with pytest.raises(ValueError, match="Invalid connection type"):
            ConnectionProperties(
                connection_type="invalid_type",
                difficulty=5,
                travel_time=30,
                description="Test",
                visibility="visible",
                requirements=[],
            )

    def test_connection_properties_defaults(self):
        """Test default values for connection properties."""
        properties = ConnectionProperties(
            connection_type="path",
            difficulty=2,
            travel_time=60,
            description="A forest path",
            visibility="visible",
            requirements=[],
        )

        assert properties.reversible is True
        assert properties.condition_flags == {}
        assert properties.special_features == []


class TestConnectionGenerationContext:
    """Test ConnectionGenerationContext data model."""

    @pytest.fixture
    def sample_location(self):
        """Create a sample location for testing."""
        return Location(
            location_id=uuid4(),
            name="Test Location",
            description="A test location",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Forest"},
        )

    def test_generation_context_creation(self, sample_location):
        """Test generation context creation."""
        source = sample_location
        target = Location(
            location_id=uuid4(),
            name="Target Location",
            description="Target for testing",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Village"},
        )

        context = ConnectionGenerationContext(
            source_location=source,
            target_location=target,
            generation_purpose="expand_world",
            distance_preference="medium",
        )

        assert context.source_location == source
        assert context.target_location == target
        assert context.generation_purpose == "expand_world"
        assert context.distance_preference == "medium"
        assert context.terrain_constraints == {}  # default
        assert context.narrative_context == {}  # default
        assert context.existing_connections == []  # default
        assert context.player_level == 1  # default

    def test_generation_context_validation(self, sample_location):
        """Test generation context validation."""
        # Test invalid generation purpose
        with pytest.raises(ValueError, match="Invalid generation purpose"):
            ConnectionGenerationContext(
                source_location=sample_location,
                target_location=sample_location,
                generation_purpose="invalid_purpose",
                distance_preference="medium",
            )

        # Test invalid distance preference
        with pytest.raises(ValueError, match="Invalid distance preference"):
            ConnectionGenerationContext(
                source_location=sample_location,
                target_location=sample_location,
                generation_purpose="expand_world",
                distance_preference="invalid_distance",
            )


class TestConnectionValidationResult:
    """Test ConnectionValidationResult data model."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ConnectionValidationResult(
            is_valid=True,
            validation_errors=[],
            warnings=["Minor issue"],
            consistency_score=0.9,
            logical_soundness=0.8,
            terrain_compatibility=0.7,
        )

        assert result.is_valid is True
        assert result.validation_errors == []
        assert result.warnings == ["Minor issue"]
        assert result.consistency_score == 0.9
        assert result.logical_soundness == 0.8
        assert result.terrain_compatibility == 0.7

    def test_validation_result_auto_invalid(self):
        """Test that validation errors automatically set is_valid to False."""
        result = ConnectionValidationResult(
            is_valid=True,  # This should be overridden
            validation_errors=["Error found"],
            warnings=[],
            consistency_score=0.5,
            logical_soundness=0.5,
            terrain_compatibility=0.5,
        )

        assert result.is_valid is False  # Automatically set due to errors

    def test_validation_result_score_validation(self):
        """Test score validation in validation result."""
        # Test invalid consistency score
        with pytest.raises(
            ValueError, match="consistency_score must be between 0.0 and 1.0"
        ):
            ConnectionValidationResult(
                is_valid=True,
                consistency_score=1.5,
                logical_soundness=0.5,
                terrain_compatibility=0.5,
            )


class TestGeneratedConnection:
    """Test GeneratedConnection data model."""

    @pytest.fixture
    def sample_properties(self):
        """Create sample connection properties."""
        return ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=45,
            description="A stone bridge",
            visibility="visible",
            requirements=[],
        )

    def test_generated_connection_creation(self, sample_properties):
        """Test generated connection creation."""
        source_id = uuid4()
        target_id = uuid4()

        connection = GeneratedConnection(
            source_location_id=source_id,
            target_location_id=target_id,
            properties=sample_properties,
        )

        assert connection.source_location_id == source_id
        assert connection.target_location_id == target_id
        assert connection.properties == sample_properties
        assert connection.metadata == {}  # default
        assert isinstance(connection.generation_timestamp, datetime)
        assert connection.embedding_vector is None  # default
        assert isinstance(connection.connection_id, UUID)

    def test_generated_connection_validation(self, sample_properties):
        """Test generated connection validation."""
        # Test same source and target
        same_id = uuid4()
        with pytest.raises(
            ValueError, match="Source and target locations cannot be the same"
        ):
            GeneratedConnection(
                source_location_id=same_id,
                target_location_id=same_id,
                properties=sample_properties,
            )


class TestConnectionArchetype:
    """Test ConnectionArchetype data model."""

    def test_archetype_creation(self):
        """Test archetype creation."""
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=45,
            description="A stone bridge",
            visibility="visible",
            requirements=[],
        )

        archetype = ConnectionArchetype(
            name="Stone Bridge",
            description="A sturdy stone bridge",
            typical_properties=properties,
            terrain_affinities={"mountain": 0.9, "river": 0.8},
            theme_compatibility={"Mountain": 0.9, "Forest": 0.7},
            generation_templates={"basic": "A bridge spans {feature}"},
        )

        assert archetype.name == "Stone Bridge"
        assert archetype.description == "A sturdy stone bridge"
        assert archetype.typical_properties == properties
        assert archetype.terrain_affinities == {"mountain": 0.9, "river": 0.8}
        assert archetype.theme_compatibility == {"Mountain": 0.9, "Forest": 0.7}
        assert archetype.rarity == "common"  # default

    def test_archetype_validation(self):
        """Test archetype validation."""
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=45,
            description="A stone bridge",
            visibility="visible",
            requirements=[],
        )

        # Test invalid rarity
        with pytest.raises(ValueError, match="Invalid rarity"):
            ConnectionArchetype(
                name="Test",
                description="Test archetype",
                typical_properties=properties,
                terrain_affinities={},
                theme_compatibility={},
                generation_templates={},
                rarity="invalid_rarity",
            )


class TestConnectionSearchCriteria:
    """Test ConnectionSearchCriteria data model."""

    def test_search_criteria_creation(self):
        """Test search criteria creation."""
        criteria = ConnectionSearchCriteria(
            connection_types=["bridge", "path"],
            source_location_themes=["Forest"],
            target_location_themes=["Mountain"],
            min_difficulty=2,
            max_difficulty=8,
            visibility_types=["visible", "partially_hidden"],
        )

        assert criteria.connection_types == ["bridge", "path"]
        assert criteria.source_location_themes == ["Forest"]
        assert criteria.target_location_themes == ["Mountain"]
        assert criteria.min_difficulty == 2
        assert criteria.max_difficulty == 8
        assert criteria.visibility_types == ["visible", "partially_hidden"]

    def test_search_criteria_validation(self):
        """Test search criteria validation."""
        # Test invalid min difficulty
        with pytest.raises(ValueError, match="Min difficulty must be between 1 and 10"):
            ConnectionSearchCriteria(min_difficulty=0)

        # Test invalid max difficulty
        with pytest.raises(ValueError, match="Max difficulty must be between 1 and 10"):
            ConnectionSearchCriteria(max_difficulty=15)

        # Test min > max
        with pytest.raises(
            ValueError, match="Min difficulty cannot exceed max difficulty"
        ):
            ConnectionSearchCriteria(min_difficulty=8, max_difficulty=5)


class TestConnectionStorageResult:
    """Test ConnectionStorageResult data model."""

    def test_storage_result_success(self):
        """Test successful storage result."""
        connection_id = uuid4()
        result = ConnectionStorageResult(
            success=True,
            connection_id=connection_id,
            storage_time_ms=150,
        )

        assert result.success is True
        assert result.connection_id == connection_id
        assert result.error_message == ""  # default
        assert result.storage_time_ms == 150

    def test_storage_result_validation(self):
        """Test storage result validation."""
        # Test successful without connection_id
        with pytest.raises(
            ValueError, match="Successful storage must include connection_id"
        ):
            ConnectionStorageResult(success=True)

        # Test failed without error message
        with pytest.raises(
            ValueError, match="Failed storage must include error_message"
        ):
            ConnectionStorageResult(success=False)


class TestConnectionMetrics:
    """Test ConnectionMetrics data model."""

    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = ConnectionMetrics(
            generation_time_ms=250,
            validation_score=0.85,
            consistency_metrics={"theme": 0.9, "terrain": 0.8},
        )

        assert metrics.generation_time_ms == 250
        assert metrics.validation_score == 0.85
        assert metrics.consistency_metrics == {"theme": 0.9, "terrain": 0.8}
        assert metrics.generation_model == "unknown"  # default
        assert metrics.template_version == "1.0"  # default
        assert metrics.context_complexity == 1  # default

    def test_metrics_validation(self):
        """Test metrics validation."""
        # Test negative generation time
        with pytest.raises(ValueError, match="Generation time cannot be negative"):
            ConnectionMetrics(
                generation_time_ms=-100,
                validation_score=0.8,
                consistency_metrics={},
            )

        # Test invalid validation score
        with pytest.raises(
            ValueError, match="Validation score must be between 0.0 and 1.0"
        ):
            ConnectionMetrics(
                generation_time_ms=100,
                validation_score=1.5,
                consistency_metrics={},
            )


class TestWorldConnectivityGraph:
    """Test WorldConnectivityGraph data model."""

    @pytest.fixture
    def sample_connection(self):
        """Create a sample connection for testing."""
        properties = ConnectionProperties(
            connection_type="bridge",
            difficulty=3,
            travel_time=45,
            description="A test bridge",
            visibility="visible",
            requirements=[],
        )

        return GeneratedConnection(
            source_location_id=uuid4(),
            target_location_id=uuid4(),
            properties=properties,
        )

    def test_graph_creation(self):
        """Test graph creation."""
        graph = WorldConnectivityGraph(nodes={}, edges={})

        assert graph.nodes == {}
        assert graph.edges == {}
        assert graph.adjacency_list == {}
        assert graph.path_cache == {}

    def test_add_connection(self, sample_connection):
        """Test adding connection to graph."""
        graph = WorldConnectivityGraph(nodes={}, edges={})
        graph.add_connection(sample_connection)

        source_id = sample_connection.source_location_id
        target_id = sample_connection.target_location_id

        # Check edges
        assert (source_id, target_id) in graph.edges
        assert graph.edges[(source_id, target_id)] == sample_connection

        # Check adjacency list
        assert source_id in graph.adjacency_list
        assert target_id in graph.adjacency_list[source_id]

        # Check reverse connection (since reversible=True by default)
        assert (target_id, source_id) in graph.edges
        assert target_id in graph.adjacency_list
        assert source_id in graph.adjacency_list[target_id]

    def test_add_non_reversible_connection(self, sample_connection):
        """Test adding non-reversible connection."""
        sample_connection.properties.reversible = False
        graph = WorldConnectivityGraph(nodes={}, edges={})
        graph.add_connection(sample_connection)

        source_id = sample_connection.source_location_id
        target_id = sample_connection.target_location_id

        # Check forward connection exists
        assert (source_id, target_id) in graph.edges

        # Check reverse connection does not exist
        assert (target_id, source_id) not in graph.edges

    def test_get_connections_from(self, sample_connection):
        """Test getting connections from a location."""
        graph = WorldConnectivityGraph(nodes={}, edges={})
        graph.add_connection(sample_connection)

        source_id = sample_connection.source_location_id
        connections = graph.get_connections_from(source_id)

        assert len(connections) == 1
        assert connections[0] == sample_connection


class TestConnectionGenerationRequest:
    """Test ConnectionGenerationRequest data model."""

    def test_generation_request_creation(self):
        """Test generation request creation."""
        source_id = uuid4()
        target_id = uuid4()

        request = ConnectionGenerationRequest(
            source_location_id=source_id,
            target_location_id=target_id,
            purpose="expand_world",
        )

        assert request.source_location_id == source_id
        assert request.target_location_id == target_id
        assert request.purpose == "expand_world"
        assert request.distance_preference == "medium"  # default
        assert request.override_properties == {}  # default
        assert request.generation_constraints == {}  # default

    def test_generation_request_validation(self):
        """Test generation request validation."""
        # Test same source and target
        same_id = uuid4()
        with pytest.raises(
            ValueError, match="Source and target locations cannot be the same"
        ):
            ConnectionGenerationRequest(
                source_location_id=same_id,
                target_location_id=same_id,
                purpose="expand_world",
            )
