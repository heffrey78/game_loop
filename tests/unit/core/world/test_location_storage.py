"""
Unit tests for LocationStorage.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import (
    CachedGeneration,
    EmbeddingUpdateResult,
    GeneratedLocation,
    LocationConnection,
    LocationTheme,
    StorageResult,
)
from game_loop.core.world.location_storage import LocationStorage
from game_loop.state.models import Location


@pytest.fixture
def sample_theme():
    """Create a sample location theme."""
    return LocationTheme(
        name="Forest",
        description="Dense woodland",
        visual_elements=["trees", "leaves"],
        atmosphere="peaceful",
        typical_objects=["log", "mushroom"],
        typical_npcs=["rabbit", "hermit"],
        generation_parameters={"complexity": "medium"},
        theme_id=uuid4(),
    )


@pytest.fixture
def sample_generated_location(sample_theme):
    """Create a sample generated location."""
    return GeneratedLocation(
        name="Mystic Grove",
        description="A beautiful grove filled with ancient trees and magical energy.",
        theme=sample_theme,
        location_type="clearing",
        objects=["ancient oak", "glowing mushroom"],
        npcs=["forest spirit"],
        connections={"east": "mountain_path"},
        metadata={"generation_time_ms": 1500},
        short_description="A mystic grove",
        atmosphere="mystical",
        special_features=["magical aura"],
    )


@pytest.fixture
def mock_session_factory():
    """Create a mock session factory."""
    session = AsyncMock()
    session_factory = Mock()
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = session
    context_manager.__aexit__.return_value = None
    session_factory.get_session.return_value = context_manager
    return session_factory, session


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    return AsyncMock()


@pytest.fixture
def location_storage(mock_session_factory, mock_embedding_manager):
    """Create a LocationStorage instance with mocked dependencies."""
    session_factory, _ = mock_session_factory
    return LocationStorage(session_factory, mock_embedding_manager)


class TestLocationStorage:
    """Test cases for LocationStorage."""

    @pytest.mark.asyncio
    async def test_store_generated_location_success(
        self,
        location_storage,
        sample_generated_location,
        mock_session_factory,
        mock_embedding_manager,
    ):
        """Test successful storage of a generated location."""
        session_factory, session = mock_session_factory

        # Mock successful database operations
        session.execute.return_value = None
        session.commit.return_value = None

        # Mock successful embedding generation
        mock_embedding_manager.create_or_update_location_embedding.return_value = True

        result = await location_storage.store_generated_location(
            sample_generated_location
        )

        assert isinstance(result, StorageResult)
        assert result.success is True
        assert result.location_id is not None
        assert result.embedding_generated is True
        assert result.storage_time_ms >= 0  # Can be 0 for very fast mock operations

        # Verify database calls
        assert session.execute.call_count >= 1  # At least location insert
        session.commit.assert_called_once()

        # Verify embedding generation was called
        mock_embedding_manager.create_or_update_location_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_generated_location_embedding_failure(
        self,
        location_storage,
        sample_generated_location,
        mock_session_factory,
        mock_embedding_manager,
    ):
        """Test location storage when embedding generation fails."""
        session_factory, session = mock_session_factory

        # Mock successful database operations
        session.execute.return_value = None
        session.commit.return_value = None

        # Mock embedding generation failure
        mock_embedding_manager.create_or_update_location_embedding.side_effect = (
            Exception("Embedding error")
        )

        result = await location_storage.store_generated_location(
            sample_generated_location
        )

        assert result.success is True  # Should still succeed
        assert result.embedding_generated is False  # But embedding failed

    @pytest.mark.asyncio
    async def test_store_generated_location_database_error(
        self, location_storage, sample_generated_location, mock_session_factory
    ):
        """Test location storage when database operation fails."""
        session_factory, session = mock_session_factory

        # Mock database error
        session.execute.side_effect = Exception("Database error")

        result = await location_storage.store_generated_location(
            sample_generated_location
        )

        assert result.success is False
        assert result.error_message == "Database error"
        assert result.location_id is None

    @pytest.mark.asyncio
    async def test_retrieve_location_from_cache(self, location_storage):
        """Test retrieving location from memory cache."""
        location_id = uuid4()
        cached_location = Location(
            location_id=location_id,
            name="Cached Location",
            description="A cached location",
            connections={},
            objects={},
            npcs={},
            state_flags={},
        )

        # Pre-populate cache
        location_storage._memory_cache[location_id] = cached_location

        result = await location_storage.retrieve_location(location_id)

        assert result is cached_location

    @pytest.mark.asyncio
    async def test_retrieve_location_from_database(
        self, location_storage, mock_session_factory
    ):
        """Test retrieving location from database."""
        session_factory, session = mock_session_factory
        location_id = uuid4()

        # Create proper UUIDs for the test
        north_location_id = uuid4()
        tree_id = uuid4()
        hermit_id = uuid4()

        # Mock database response with valid JSON structure
        mock_row = Mock()
        mock_row.location_id = location_id
        mock_row.name = "Database Location"
        mock_row.description = "A location from database"
        mock_row.connections = f'{{"north": "{north_location_id}"}}'
        mock_row.objects = f'{{"{tree_id}": {{"object_id": "{tree_id}", "name": "tree", "description": "A tall tree", "is_takeable": false}}}}'
        mock_row.npcs = f'{{"{hermit_id}": {{"npc_id": "{hermit_id}", "name": "hermit", "description": "A wise hermit", "dialogue_state": "neutral"}}}}'
        mock_row.state_flags = '{"visit_count": 1}'
        mock_row.generation_metadata = '{"type": "clearing"}'
        mock_row.theme_name = "Forest"

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        session.execute.return_value = mock_result

        result = await location_storage.retrieve_location(location_id)

        assert result is not None
        assert result.name == "Database Location"
        assert result.location_id == location_id
        assert "north" in result.connections
        assert tree_id in result.objects
        assert hermit_id in result.npcs
        assert result.state_flags["theme"] == "Forest"

        # Should be cached after retrieval
        assert location_id in location_storage._memory_cache

    @pytest.mark.asyncio
    async def test_retrieve_location_not_found(
        self, location_storage, mock_session_factory
    ):
        """Test retrieving non-existent location."""
        session_factory, session = mock_session_factory
        location_id = uuid4()

        # Mock empty database response
        mock_result = Mock()
        mock_result.fetchone.return_value = None
        session.execute.return_value = mock_result

        result = await location_storage.retrieve_location(location_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_location(self, location_storage):
        """Test caching a location."""
        location = Location(
            location_id=uuid4(),
            name="Test Location",
            description="Test description",
            connections={},
            objects={},
            npcs={},
            state_flags={},
        )

        cache_duration = timedelta(hours=1)

        await location_storage.cache_location(location, cache_duration)

        assert location.location_id in location_storage._memory_cache
        assert location_storage._memory_cache[location.location_id] is location

    @pytest.mark.asyncio
    async def test_update_location_embeddings_success(
        self, location_storage, mock_embedding_manager
    ):
        """Test successful embedding update."""
        location_id = uuid4()

        # Mock successful embedding update
        mock_embedding_manager.create_or_update_location_embedding.return_value = True

        result = await location_storage.update_location_embeddings(location_id)

        assert isinstance(result, EmbeddingUpdateResult)
        assert result.success is True
        assert result.location_id == location_id
        assert result.update_time_ms >= 0  # Can be 0 for very fast mock operations

        mock_embedding_manager.create_or_update_location_embedding.assert_called_once_with(
            location_id
        )

    @pytest.mark.asyncio
    async def test_update_location_embeddings_failure(
        self, location_storage, mock_embedding_manager
    ):
        """Test embedding update failure."""
        location_id = uuid4()

        # Mock embedding update failure
        mock_embedding_manager.create_or_update_location_embedding.side_effect = (
            Exception("Embedding error")
        )

        result = await location_storage.update_location_embeddings(location_id)

        assert result.success is False
        assert result.error_message == "Embedding error"

    @pytest.mark.asyncio
    async def test_store_location_connections(
        self, location_storage, mock_session_factory
    ):
        """Test storing location connections."""
        session_factory, session = mock_session_factory

        # Mock successful database operations
        session.execute.return_value = None
        session.commit.return_value = None

        connections = [
            LocationConnection(
                from_location_id=uuid4(),
                to_location_id=uuid4(),
                direction="north",
                connection_type="normal",
                description="Path north",
                is_bidirectional=True,
            ),
            LocationConnection(
                from_location_id=uuid4(),
                to_location_id=uuid4(),
                direction="south",
                connection_type="normal",
                description="Path south",
                is_bidirectional=False,
            ),
        ]

        result = await location_storage.store_location_connections(connections)

        assert result is True

        # Should update connections for bidirectional (first) and unidirectional (second)
        # First connection: 2 updates (from->to, to->from)
        # Second connection: 1 update (from->to only)
        assert session.execute.call_count == 3
        session.commit.assert_called_once()

    def test_get_reverse_direction(self, location_storage):
        """Test getting reverse directions."""
        assert location_storage._get_reverse_direction("north") == "south"
        assert location_storage._get_reverse_direction("south") == "north"
        assert location_storage._get_reverse_direction("east") == "west"
        assert location_storage._get_reverse_direction("west") == "east"
        assert location_storage._get_reverse_direction("up") == "down"
        assert location_storage._get_reverse_direction("down") == "up"
        assert location_storage._get_reverse_direction("in") == "out"
        assert location_storage._get_reverse_direction("out") == "in"
        assert location_storage._get_reverse_direction("unknown") is None

    @pytest.mark.asyncio
    async def test_cache_generation_result(
        self, location_storage, sample_generated_location, mock_session_factory
    ):
        """Test caching generation result."""
        session_factory, session = mock_session_factory

        # Mock successful database operations
        session.execute.return_value = None
        session.commit.return_value = None

        context_hash = "test_hash_123"
        cache_duration = timedelta(hours=1)

        await location_storage.cache_generation_result(
            context_hash, sample_generated_location, cache_duration
        )

        # Should be in memory cache
        assert context_hash in location_storage._generation_cache
        cached = location_storage._generation_cache[context_hash]
        assert cached.generated_location == sample_generated_location
        assert not cached.is_expired

        # Should also attempt database storage
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_generation_from_memory(
        self, location_storage, sample_generated_location
    ):
        """Test getting cached generation from memory."""
        context_hash = "test_hash_123"

        # Pre-populate memory cache
        cached_generation = CachedGeneration(
            context_hash=context_hash,
            generated_location=sample_generated_location,
            cache_expires_at=datetime.now() + timedelta(hours=1),
            usage_count=0,
        )
        location_storage._generation_cache[context_hash] = cached_generation

        result = await location_storage.get_cached_generation(context_hash)

        assert result is cached_generation
        assert result.usage_count == 1  # Should increment usage

    @pytest.mark.asyncio
    async def test_get_cached_generation_expired(
        self, location_storage, sample_generated_location
    ):
        """Test getting expired cached generation."""
        context_hash = "test_hash_123"

        # Pre-populate memory cache with expired entry
        cached_generation = CachedGeneration(
            context_hash=context_hash,
            generated_location=sample_generated_location,
            cache_expires_at=datetime.now() - timedelta(hours=1),  # Expired
            usage_count=0,
        )
        location_storage._generation_cache[context_hash] = cached_generation

        result = await location_storage.get_cached_generation(context_hash)

        assert result is None
        assert (
            context_hash not in location_storage._generation_cache
        )  # Should be removed

    @pytest.mark.asyncio
    async def test_get_cached_generation_from_database(
        self, location_storage, mock_session_factory, sample_theme
    ):
        """Test getting cached generation from database."""
        session_factory, session = mock_session_factory
        context_hash = "test_hash_123"

        # Mock database response
        mock_row = Mock()
        mock_row.generated_location = json.dumps(
            {
                "name": "Cached Location",
                "description": "From database cache",
                "theme_name": "Forest",
                "location_type": "clearing",
                "objects": ["cached_object"],
                "npcs": ["cached_npc"],
                "connections": {},
                "metadata": {},
            }
        )
        mock_row.cache_expires_at = datetime.now() + timedelta(hours=1)
        mock_row.usage_count = 0
        mock_row.created_at = datetime.now()

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        session.execute.return_value = mock_result

        result = await location_storage.get_cached_generation(context_hash)

        assert result is not None
        assert result.generated_location.name == "Cached Location"
        assert result.context_hash == context_hash

        # Should update usage count in database
        assert session.execute.call_count == 2  # Select + Update
        session.commit.assert_called_once()

        # Should be cached in memory
        assert context_hash in location_storage._generation_cache

    def test_generate_context_hash(self, location_storage):
        """Test generation of context hash."""
        context_data = {
            "expansion_point": {"location_id": uuid4(), "direction": "north"},
            "adjacent_locations": [{"name": "Location A"}, {"name": "Location B"}],
            "world_themes": ["Forest", "Village"],
            "player_preferences": {"complexity": "medium"},
        }

        result = location_storage.generate_context_hash(context_data)

        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest length

        # Same input should produce same hash
        result2 = location_storage.generate_context_hash(context_data)
        assert result == result2

    @pytest.mark.asyncio
    async def test_cleanup_expired_cache(
        self, location_storage, mock_session_factory, sample_generated_location
    ):
        """Test cleanup of expired cache entries."""
        session_factory, session = mock_session_factory

        # Add expired entry to memory cache
        expired_cached = CachedGeneration(
            context_hash="expired_hash",
            generated_location=sample_generated_location,
            cache_expires_at=datetime.now() - timedelta(hours=1),  # Expired
            usage_count=0,
        )
        location_storage._generation_cache["expired_hash"] = expired_cached

        # Add valid entry to memory cache
        valid_cached = CachedGeneration(
            context_hash="valid_hash",
            generated_location=sample_generated_location,
            cache_expires_at=datetime.now() + timedelta(hours=1),  # Valid
            usage_count=0,
        )
        location_storage._generation_cache["valid_hash"] = valid_cached

        # Mock database cleanup (1 row deleted)
        mock_result = Mock()
        mock_result.rowcount = 1
        session.execute.return_value = mock_result
        session.commit.return_value = None

        removed_count = await location_storage.cleanup_expired_cache()

        # Should remove 1 from memory + 1 from database = 2 total
        assert removed_count == 2

        # Expired entry should be removed from memory
        assert "expired_hash" not in location_storage._generation_cache

        # Valid entry should remain
        assert "valid_hash" in location_storage._generation_cache

    def test_clear_cache(self, location_storage):
        """Test clearing all caches."""
        # Populate caches
        location_storage._memory_cache[uuid4()] = Mock()
        location_storage._generation_cache["test_hash"] = Mock()

        location_storage.clear_cache()

        assert len(location_storage._memory_cache) == 0
        assert len(location_storage._generation_cache) == 0


class TestCachedGeneration:
    """Test cases for CachedGeneration model."""

    def test_is_expired_true(self, sample_generated_location):
        """Test expired cache detection."""
        cached = CachedGeneration(
            context_hash="test",
            generated_location=sample_generated_location,
            cache_expires_at=datetime.now() - timedelta(hours=1),  # Past
        )

        assert cached.is_expired is True

    def test_is_expired_false(self, sample_generated_location):
        """Test non-expired cache detection."""
        cached = CachedGeneration(
            context_hash="test",
            generated_location=sample_generated_location,
            cache_expires_at=datetime.now() + timedelta(hours=1),  # Future
        )

        assert cached.is_expired is False

    def test_time_until_expiry(self, sample_generated_location):
        """Test time until expiry calculation."""
        future_time = datetime.now() + timedelta(hours=2)
        cached = CachedGeneration(
            context_hash="test",
            generated_location=sample_generated_location,
            cache_expires_at=future_time,
        )

        time_until = cached.time_until_expiry

        assert isinstance(time_until, timedelta)
        assert time_until.total_seconds() > 0
        assert time_until.total_seconds() <= 2 * 3600  # Less than or equal to 2 hours
