"""
Unit tests for object models.

Tests all object model data structures, validation, and relationships.
"""

from datetime import datetime
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import LocationTheme
from game_loop.core.models.object_models import (
    GeneratedObject,
    GenerationMetrics,
    ObjectArchetype,
    ObjectGenerationContext,
    ObjectGenerationRequest,
    ObjectInteractions,
    ObjectPlacement,
    ObjectProperties,
    ObjectSearchCriteria,
    ObjectStorageResult,
    ObjectTheme,
    ObjectValidationResult,
)
from game_loop.state.models import Location, WorldObject


class TestObjectProperties:
    """Tests for ObjectProperties model."""

    def test_object_properties_creation(self):
        """Test basic object properties creation."""
        props = ObjectProperties(
            name="Iron Sword",
            object_type="weapon",
            material="iron",
            size="medium",
            weight="heavy",
        )

        assert props.name == "Iron Sword"
        assert props.object_type == "weapon"
        assert props.material == "iron"
        assert props.size == "medium"
        assert props.weight == "heavy"
        assert props.durability == "sturdy"  # Default value
        assert props.value == 0  # Default value
        assert props.special_properties == []  # Default value

    def test_object_properties_with_all_fields(self):
        """Test object properties with all fields specified."""
        props = ObjectProperties(
            name="Masterwork Blade",
            object_type="weapon",
            material="enchanted_steel",
            size="medium",
            weight="normal",
            durability="very_sturdy",
            value=500,
            special_properties=["sharp", "magical", "masterwork"],
            cultural_significance="legendary",
            description="A blade forged by master smiths.",
        )

        assert props.name == "Masterwork Blade"
        assert props.value == 500
        assert "magical" in props.special_properties
        assert props.cultural_significance == "legendary"
        assert props.description == "A blade forged by master smiths."

    def test_object_properties_validation(self):
        """Test object properties validation."""
        # Test empty name validation
        with pytest.raises(ValueError, match="Object name cannot be empty"):
            ObjectProperties(name="", object_type="weapon")

        # Test empty object type validation
        with pytest.raises(ValueError, match="Object type cannot be empty"):
            ObjectProperties(name="Test", object_type="")


class TestObjectInteractions:
    """Tests for ObjectInteractions model."""

    def test_object_interactions_creation(self):
        """Test basic object interactions creation."""
        interactions = ObjectInteractions(
            available_actions=["examine", "take", "use"],
            portable=True,
            consumable=False,
        )

        assert "examine" in interactions.available_actions
        assert interactions.portable is True
        assert interactions.consumable is False
        assert interactions.use_requirements == {}  # Default

    def test_object_interactions_default_actions(self):
        """Test that default actions are set when none provided."""
        interactions = ObjectInteractions()

        assert "examine" in interactions.available_actions
        assert "take" in interactions.available_actions
        assert "drop" in interactions.available_actions
        assert interactions.portable is True  # Default

    def test_object_interactions_non_portable(self):
        """Test interactions for non-portable objects."""
        interactions = ObjectInteractions(portable=False)

        assert "examine" in interactions.available_actions
        assert "take" not in interactions.available_actions
        assert "drop" not in interactions.available_actions

    def test_object_interactions_comprehensive(self):
        """Test comprehensive object interactions."""
        interactions = ObjectInteractions(
            available_actions=["examine", "read", "study"],
            use_requirements={"read": "literacy"},
            interaction_results={"read": "You read the ancient text."},
            state_changes={"study": "knowledge_gained"},
            consumable=False,
            portable=True,
            examination_text="An old tome with mysterious symbols.",
            hidden_properties={"secret": "hidden_knowledge"},
        )

        assert "study" in interactions.available_actions
        assert interactions.use_requirements["read"] == "literacy"
        assert "knowledge_gained" in interactions.state_changes.values()
        assert interactions.hidden_properties["secret"] == "hidden_knowledge"


class TestObjectGenerationContext:
    """Tests for ObjectGenerationContext model."""

    def create_test_location(self) -> Location:
        """Create a test location."""
        return Location(
            location_id=uuid4(),
            name="Test Village",
            description="A small test village",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Village"},
        )

    def create_test_theme(self) -> LocationTheme:
        """Create a test location theme."""
        return LocationTheme(
            name="Village",
            description="Rural village setting",
            visual_elements=["cobblestone", "thatched_roofs"],
            atmosphere="peaceful",
            typical_objects=["tool", "container", "furniture"],
            typical_npcs=["villager", "merchant"],
            generation_parameters={"object_affinities": {"tool": 0.8}},
        )

    def test_generation_context_creation(self):
        """Test basic generation context creation."""
        location = self.create_test_location()
        theme = self.create_test_theme()

        context = ObjectGenerationContext(
            location=location,
            location_theme=theme,
            generation_purpose="populate_location",
            existing_objects=[],
            player_level=3,
        )

        assert context.location.name == "Test Village"
        assert context.location_theme.name == "Village"
        assert context.generation_purpose == "populate_location"
        assert context.player_level == 3
        assert context.existing_objects == []

    def test_generation_context_validation(self):
        """Test generation context validation."""
        location = self.create_test_location()
        theme = self.create_test_theme()

        # Test invalid player level
        with pytest.raises(ValueError, match="Player level must be at least 1"):
            ObjectGenerationContext(
                location=location,
                location_theme=theme,
                generation_purpose="populate_location",
                existing_objects=[],
                player_level=0,
            )

        # Test invalid generation purpose
        with pytest.raises(ValueError, match="Invalid generation purpose"):
            ObjectGenerationContext(
                location=location,
                location_theme=theme,
                generation_purpose="invalid_purpose",
                existing_objects=[],
                player_level=3,
            )


class TestObjectPlacement:
    """Tests for ObjectPlacement model."""

    def test_object_placement_creation(self):
        """Test basic object placement creation."""
        object_id = uuid4()
        location_id = uuid4()

        placement = ObjectPlacement(
            object_id=object_id,
            location_id=location_id,
            placement_type="floor",
            visibility="visible",
            accessibility="accessible",
            spatial_description="On the wooden floor",
            discovery_difficulty=2,
        )

        assert placement.object_id == object_id
        assert placement.location_id == location_id
        assert placement.placement_type == "floor"
        assert placement.discovery_difficulty == 2

    def test_object_placement_validation(self):
        """Test object placement validation."""
        object_id = uuid4()
        location_id = uuid4()

        # Test invalid placement type
        with pytest.raises(ValueError, match="Invalid placement type"):
            ObjectPlacement(
                object_id=object_id,
                location_id=location_id,
                placement_type="invalid_type",
            )

        # Test invalid discovery difficulty
        with pytest.raises(
            ValueError, match="Discovery difficulty must be between 1 and 10"
        ):
            ObjectPlacement(
                object_id=object_id,
                location_id=location_id,
                placement_type="floor",
                discovery_difficulty=15,
            )


class TestGeneratedObject:
    """Tests for GeneratedObject model."""

    def create_test_objects(self):
        """Create test objects for generated object."""
        base_object = WorldObject(
            object_id=uuid4(), name="Test Sword", description="A basic sword"
        )

        properties = ObjectProperties(
            name="Test Sword", object_type="weapon", material="iron"
        )

        interactions = ObjectInteractions(available_actions=["examine", "wield"])

        return base_object, properties, interactions

    def test_generated_object_creation(self):
        """Test basic generated object creation."""
        base_object, properties, interactions = self.create_test_objects()

        generated_object = GeneratedObject(
            base_object=base_object,
            properties=properties,
            interactions=interactions,
            generation_metadata={"test": "data"},
            embedding_vector=[0.1, 0.2, 0.3],
        )

        assert generated_object.base_object.name == "Test Sword"
        assert generated_object.properties.object_type == "weapon"
        assert "examine" in generated_object.interactions.available_actions
        assert generated_object.generation_metadata["test"] == "data"
        assert len(generated_object.embedding_vector) == 3

    def test_generated_object_consistency(self):
        """Test that generated object maintains consistency."""
        base_object, properties, interactions = self.create_test_objects()

        # Different names should be synchronized
        properties.name = "Enhanced Sword"

        generated_object = GeneratedObject(
            base_object=base_object, properties=properties, interactions=interactions
        )

        # Should synchronize names
        assert generated_object.base_object.name == "Enhanced Sword"


class TestObjectArchetype:
    """Tests for ObjectArchetype model."""

    def test_object_archetype_creation(self):
        """Test basic object archetype creation."""
        properties = ObjectProperties(name="Basic Sword", object_type="weapon")

        archetype = ObjectArchetype(
            name="sword",
            description="A bladed weapon",
            typical_properties=properties,
            location_affinities={"Village": 0.6, "City": 0.8},
            rarity="uncommon",
        )

        assert archetype.name == "sword"
        assert archetype.description == "A bladed weapon"
        assert archetype.typical_properties.object_type == "weapon"
        assert archetype.location_affinities["City"] == 0.8
        assert archetype.rarity == "uncommon"

    def test_object_archetype_validation(self):
        """Test object archetype validation."""
        properties = ObjectProperties(name="Test", object_type="weapon")

        # Test empty name
        with pytest.raises(ValueError, match="Archetype name cannot be empty"):
            ObjectArchetype(name="", description="Test", typical_properties=properties)

        # Test invalid rarity
        with pytest.raises(ValueError, match="Invalid rarity"):
            ObjectArchetype(
                name="test",
                description="Test",
                typical_properties=properties,
                rarity="invalid_rarity",
            )


class TestObjectSearchCriteria:
    """Tests for ObjectSearchCriteria model."""

    def test_search_criteria_creation(self):
        """Test basic search criteria creation."""
        criteria = ObjectSearchCriteria(
            query_text="sharp weapon",
            object_types=["weapon"],
            max_results=5,
            similarity_threshold=0.8,
        )

        assert criteria.query_text == "sharp weapon"
        assert "weapon" in criteria.object_types
        assert criteria.max_results == 5
        assert criteria.similarity_threshold == 0.8

    def test_search_criteria_validation(self):
        """Test search criteria validation."""
        # Test invalid max_results
        with pytest.raises(ValueError, match="Max results must be positive"):
            ObjectSearchCriteria(max_results=0)

        # Test invalid similarity_threshold
        with pytest.raises(
            ValueError, match="Similarity threshold must be between 0.0 and 1.0"
        ):
            ObjectSearchCriteria(similarity_threshold=1.5)


class TestObjectValidationResult:
    """Tests for ObjectValidationResult model."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ObjectValidationResult(
            is_valid=True,
            validation_errors=[],
            warnings=["Minor issue"],
            consistency_score=0.9,
        )

        assert result.is_valid is True
        assert len(result.validation_errors) == 0
        assert len(result.warnings) == 1
        assert result.consistency_score == 0.9

    def test_validation_result_auto_invalid(self):
        """Test that validation result becomes invalid with errors."""
        result = ObjectValidationResult(
            is_valid=True,  # This should be overridden
            validation_errors=["Error 1", "Error 2"],
        )

        assert result.is_valid is False  # Auto-set due to errors


class TestObjectStorageResult:
    """Tests for ObjectStorageResult model."""

    def test_storage_result_success(self):
        """Test successful storage result."""
        object_id = uuid4()
        result = ObjectStorageResult(
            success=True,
            object_id=object_id,
            embedding_generated=True,
            storage_duration_ms=150.5,
        )

        assert result.success is True
        assert result.object_id == object_id
        assert result.embedding_generated is True
        assert result.storage_duration_ms == 150.5

    def test_storage_result_validation(self):
        """Test storage result validation."""
        # Test missing object_id on successful storage
        with pytest.raises(
            ValueError, match="Successful storage must include object_id"
        ):
            ObjectStorageResult(success=True, object_id=None)


class TestGenerationMetrics:
    """Tests for GenerationMetrics model."""

    def test_generation_metrics_creation(self):
        """Test generation metrics creation."""
        metrics = GenerationMetrics()

        assert isinstance(metrics.generation_start_time, datetime)
        assert metrics.generation_end_time is None
        assert metrics.total_time_ms == 0.0
        assert metrics.cache_hit is False

    def test_generation_metrics_completion(self):
        """Test generation metrics completion."""
        metrics = GenerationMetrics()
        metrics.llm_generation_time_ms = 100.0
        metrics.validation_time_ms = 50.0

        # Mark as complete
        metrics.mark_complete()

        assert metrics.generation_end_time is not None
        assert metrics.total_time_ms > 0.0


class TestObjectTheme:
    """Tests for ObjectTheme model."""

    def test_object_theme_creation(self):
        """Test object theme creation."""
        theme = ObjectTheme(
            name="Village",
            description="Rural village setting",
            typical_materials=["wood", "iron"],
            common_object_types=["tool", "furniture"],
            style_descriptors=["rustic", "practical"],
        )

        assert theme.name == "Village"
        assert "wood" in theme.typical_materials
        assert "tool" in theme.common_object_types
        assert "rustic" in theme.style_descriptors

    def test_object_theme_validation(self):
        """Test object theme validation."""
        with pytest.raises(ValueError, match="Theme name cannot be empty"):
            ObjectTheme(name="", description="Test")


class TestObjectGenerationRequest:
    """Tests for ObjectGenerationRequest model."""

    def test_generation_request_creation(self):
        """Test generation request creation."""
        location_id = uuid4()
        request = ObjectGenerationRequest(
            location_id=location_id,
            purpose="populate_location",
            quantity=3,
            priority="high",
        )

        assert request.location_id == location_id
        assert request.purpose == "populate_location"
        assert request.quantity == 3
        assert request.priority == "high"

    def test_generation_request_validation(self):
        """Test generation request validation."""
        location_id = uuid4()

        # Test invalid quantity
        with pytest.raises(ValueError, match="Quantity must be positive"):
            ObjectGenerationRequest(location_id=location_id, purpose="test", quantity=0)

        # Test invalid priority
        with pytest.raises(ValueError, match="Invalid priority"):
            ObjectGenerationRequest(
                location_id=location_id, purpose="test", priority="invalid_priority"
            )
