"""
Unit tests for Connection Theme Manager.

Tests connection theme management, archetype handling, and terrain compatibility.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from game_loop.core.models.connection_models import (
    ConnectionArchetype,
    ConnectionGenerationContext,
)
from game_loop.core.world.connection_theme_manager import ConnectionThemeManager
from game_loop.state.models import Location, WorldState


class TestConnectionThemeManager:
    """Test ConnectionThemeManager functionality."""

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
        """Create ConnectionThemeManager instance."""
        return ConnectionThemeManager(world_state, session_factory)

    @pytest.fixture
    def forest_location(self):
        """Create forest location for testing."""
        return Location(
            location_id=uuid4(),
            name="Whispering Woods",
            description="An ancient forest",
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
    def mountain_location(self):
        """Create mountain location for testing."""
        return Location(
            location_id=uuid4(),
            name="Stormwind Peaks",
            description="High mountain peaks",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Mountain"},
        )

    @pytest.fixture
    def generation_context(self, forest_location, village_location):
        """Create test generation context."""
        return ConnectionGenerationContext(
            source_location=forest_location,
            target_location=village_location,
            generation_purpose="expand_world",
            distance_preference="medium",
        )

    @pytest.mark.asyncio
    async def test_determine_connection_type_expand_world(
        self, theme_manager, generation_context
    ):
        """Test connection type determination for world expansion."""
        connection_type = await theme_manager.determine_connection_type(
            generation_context
        )

        assert isinstance(connection_type, str)
        # Should prefer basic connection types for world expansion
        assert connection_type in ["passage", "path", "road", "bridge"]

    @pytest.mark.asyncio
    async def test_determine_connection_type_quest_path(
        self, theme_manager, generation_context
    ):
        """Test connection type determination for quest paths."""
        generation_context.generation_purpose = "quest_path"

        connection_type = await theme_manager.determine_connection_type(
            generation_context
        )

        assert isinstance(connection_type, str)
        # Quest paths might prefer special connection types
        assert connection_type in ["portal", "bridge", "gateway", "passage", "path"]

    @pytest.mark.asyncio
    async def test_get_available_connection_types_forest_village(self, theme_manager):
        """Test getting available connection types between forest and village."""
        available_types = await theme_manager.get_available_connection_types(
            "Forest", "Village"
        )

        assert isinstance(available_types, list)
        assert len(available_types) > 0
        # Forest to village should support natural connection types
        assert any(
            conn_type in ["path", "bridge", "passage"] for conn_type in available_types
        )

    @pytest.mark.asyncio
    async def test_get_available_connection_types_mountain_mountain(
        self, theme_manager
    ):
        """Test getting available connection types between mountains."""
        available_types = await theme_manager.get_available_connection_types(
            "Mountain", "Mountain"
        )

        assert isinstance(available_types, list)
        assert len(available_types) > 0
        # Mountain to mountain should support mountain-appropriate types
        assert any(
            conn_type in ["path", "bridge", "tunnel", "passage"]
            for conn_type in available_types
        )

    @pytest.mark.asyncio
    async def test_get_available_connection_types_unknown_theme(self, theme_manager):
        """Test fallback for unknown themes."""
        available_types = await theme_manager.get_available_connection_types(
            "Unknown", "Unknown"
        )

        assert isinstance(available_types, list)
        assert len(available_types) > 0
        # Should fallback to basic types
        assert "passage" in available_types or "path" in available_types

    @pytest.mark.asyncio
    async def test_get_terrain_compatibility_high(self, theme_manager):
        """Test high terrain compatibility."""
        compatibility = await theme_manager.get_terrain_compatibility(
            "forest", "forest"
        )

        assert isinstance(compatibility, float)
        assert (
            0.8 <= compatibility <= 1.0
        )  # Same terrain should have high compatibility

    @pytest.mark.asyncio
    async def test_get_terrain_compatibility_medium(self, theme_manager):
        """Test medium terrain compatibility."""
        compatibility = await theme_manager.get_terrain_compatibility(
            "forest", "grassland"
        )

        assert isinstance(compatibility, float)
        assert (
            0.3 <= compatibility <= 0.9
        )  # Compatible terrains should have medium+ compatibility

    @pytest.mark.asyncio
    async def test_get_terrain_compatibility_unknown(self, theme_manager):
        """Test terrain compatibility with unknown terrains."""
        compatibility = await theme_manager.get_terrain_compatibility(
            "unknown1", "unknown2"
        )

        assert isinstance(compatibility, float)
        assert 0.0 <= compatibility <= 1.0  # Should return valid score

    @pytest.mark.asyncio
    async def test_generate_theme_appropriate_description_bridge(
        self, theme_manager, generation_context
    ):
        """Test generating theme-appropriate description for bridge."""
        description = await theme_manager.generate_theme_appropriate_description(
            "bridge", generation_context
        )

        assert isinstance(description, str)
        assert len(description) > 0
        assert "bridge" in description.lower()

    @pytest.mark.asyncio
    async def test_generate_theme_appropriate_description_path(
        self, theme_manager, generation_context
    ):
        """Test generating theme-appropriate description for path."""
        description = await theme_manager.generate_theme_appropriate_description(
            "path", generation_context
        )

        assert isinstance(description, str)
        assert len(description) > 0
        assert "path" in description.lower()

    @pytest.mark.asyncio
    async def test_generate_theme_appropriate_description_unknown_type(
        self, theme_manager, generation_context
    ):
        """Test generating description for unknown connection type."""
        description = await theme_manager.generate_theme_appropriate_description(
            "unknown_type", generation_context
        )

        assert isinstance(description, str)
        assert len(description) > 0
        # Should include the connection type even if unknown
        assert "unknown_type" in description.lower()

    def test_get_connection_archetype_existing(self, theme_manager):
        """Test getting existing connection archetype."""
        archetype = theme_manager.get_connection_archetype("passage")

        assert isinstance(archetype, ConnectionArchetype)
        assert archetype.name == "passage"
        assert archetype.typical_properties.connection_type == "passage"

    def test_get_connection_archetype_bridge(self, theme_manager):
        """Test getting bridge archetype."""
        archetype = theme_manager.get_connection_archetype("bridge")

        assert isinstance(archetype, ConnectionArchetype)
        assert archetype.name == "bridge"
        assert archetype.typical_properties.connection_type == "bridge"
        assert archetype.typical_properties.difficulty >= 1
        assert archetype.typical_properties.travel_time > 0

    def test_get_connection_archetype_nonexistent(self, theme_manager):
        """Test getting nonexistent archetype."""
        archetype = theme_manager.get_connection_archetype("nonexistent")

        assert archetype is None

    def test_archetype_initialization(self, theme_manager):
        """Test that default archetypes are properly initialized."""
        # Check that common archetypes exist
        passage_archetype = theme_manager.get_connection_archetype("passage")
        bridge_archetype = theme_manager.get_connection_archetype("bridge")
        portal_archetype = theme_manager.get_connection_archetype("portal")
        path_archetype = theme_manager.get_connection_archetype("path")

        assert passage_archetype is not None
        assert bridge_archetype is not None
        assert portal_archetype is not None
        assert path_archetype is not None

        # Check archetype properties
        assert passage_archetype.rarity == "common"
        assert portal_archetype.rarity == "rare"

        # Check terrain affinities exist
        assert len(bridge_archetype.terrain_affinities) > 0
        assert len(portal_archetype.terrain_affinities) > 0

    def test_archetype_theme_compatibility(self, theme_manager):
        """Test archetype theme compatibility."""
        passage_archetype = theme_manager.get_connection_archetype("passage")

        assert isinstance(passage_archetype.theme_compatibility, dict)
        # Passages should be compatible with dungeon/cave themes
        assert "Dungeon" in passage_archetype.theme_compatibility
        assert passage_archetype.theme_compatibility["Dungeon"] > 0.5

    def test_archetype_terrain_affinities(self, theme_manager):
        """Test archetype terrain affinities."""
        bridge_archetype = theme_manager.get_connection_archetype("bridge")

        assert isinstance(bridge_archetype.terrain_affinities, dict)
        # Bridges should have affinity for water/river terrain
        assert "river" in bridge_archetype.terrain_affinities
        assert bridge_archetype.terrain_affinities["river"] > 0.5

    @pytest.mark.asyncio
    async def test_error_handling_in_determine_connection_type(self, theme_manager):
        """Test error handling in connection type determination."""
        # Create invalid context that might cause errors
        invalid_location = Location(
            location_id=uuid4(),
            name="Invalid",
            description="",
            connections={},
            objects={},
            npcs={},
            state_flags={},  # No theme
        )

        context = ConnectionGenerationContext(
            source_location=invalid_location,
            target_location=invalid_location,
            generation_purpose="expand_world",
            distance_preference="medium",
        )

        # Should not raise exception, should return fallback
        connection_type = await theme_manager.determine_connection_type(context)
        assert isinstance(connection_type, str)
        assert connection_type == "passage"  # Expected fallback

    @pytest.mark.asyncio
    async def test_context_based_selection(
        self, theme_manager, forest_location, mountain_location
    ):
        """Test that connection selection considers context."""
        forest_to_mountain_context = ConnectionGenerationContext(
            source_location=forest_location,
            target_location=mountain_location,
            generation_purpose="exploration",
            distance_preference="long",
            terrain_constraints={
                "source_terrain": "forest",
                "target_terrain": "mountain",
            },
        )

        connection_type = await theme_manager.determine_connection_type(
            forest_to_mountain_context
        )

        assert isinstance(connection_type, str)
        # Forest to mountain with exploration purpose should prefer challenging types
        assert connection_type in ["path", "bridge", "tunnel", "passage"]

    @pytest.mark.asyncio
    async def test_distance_preference_consideration(
        self, theme_manager, generation_context
    ):
        """Test that distance preference influences selection."""
        # Test short distance preference
        generation_context.distance_preference = "short"
        short_type = await theme_manager.determine_connection_type(generation_context)

        # Test long distance preference
        generation_context.distance_preference = "long"
        long_type = await theme_manager.determine_connection_type(generation_context)

        assert isinstance(short_type, str)
        assert isinstance(long_type, str)
        # Both should be valid connection types
        valid_types = ["passage", "bridge", "portal", "path", "tunnel", "road"]
        assert short_type in valid_types
        assert long_type in valid_types
