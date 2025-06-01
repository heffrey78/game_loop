"""
Tests for entity embedding generator functionality.

This module contains unit tests for the EntityEmbeddingGenerator
and related entity-specific embedding components.
"""

from unittest.mock import MagicMock

import pytest

from game_loop.embeddings.entity_embeddings import EntityEmbeddingGenerator


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock_service = MagicMock()

    # Create an async mock for generate_embedding
    async def mock_generate_embedding(*args, **kwargs):
        return [0.1] * 384

    # Create an async mock for generate_embeddings_batch
    async def mock_generate_embeddings_batch(*args, **kwargs):
        return [[0.1] * 384, [0.1] * 384]

    # Assign the mock methods
    mock_service.generate_embedding = MagicMock(side_effect=mock_generate_embedding)
    mock_service.generate_embeddings_batch = MagicMock(
        side_effect=mock_generate_embeddings_batch
    )

    return mock_service


@pytest.fixture
def entity_generator(mock_embedding_service):
    """Create an EntityEmbeddingGenerator instance with mock service."""
    return EntityEmbeddingGenerator(mock_embedding_service)


@pytest.mark.asyncio
async def test_generate_entity_embedding(entity_generator):
    """Test generating an embedding for a single entity."""
    # Create a test character entity
    character = {
        "id": "char1",
        "name": "Test Character",
        "description": "A character for testing",
        "personality": "Friendly and helpful",
        "background": "Created for unit tests",
    }

    # Generate embedding
    embedding = await entity_generator.generate_entity_embedding(character, "character")

    # Check if embedding has expected format
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)

    # Verify the mock was called with preprocessed text
    entity_generator.embedding_service.generate_embedding.assert_called_once()

    # The first arg should be the preprocessed text
    call_args = entity_generator.embedding_service.generate_embedding.call_args[0]
    assert len(call_args) >= 1
    assert isinstance(call_args[0], str)
    assert "Test Character" in call_args[0]


@pytest.mark.asyncio
async def test_generate_entity_embeddings_batch(entity_generator):
    """Test generating embeddings for multiple entities in a batch."""
    # Create test entities
    entities = [
        {
            "id": "char1",
            "name": "Character 1",
            "description": "The first test character",
            "type": "character",
        },
        {
            "id": "loc1",
            "name": "Location 1",
            "description": "A test location",
            "type": "location",
        },
    ]

    # Generate embeddings
    embeddings = await entity_generator.generate_entity_embeddings_batch(
        entities, entity_types=["character", "location"]
    )

    # Check if embeddings have expected format
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)

    # Verify the mock was called
    entity_generator.embedding_service.generate_embeddings_batch.assert_called_once()


def test_preprocess_entity(entity_generator):
    """Test preprocessing an entity to text."""
    # Create a test item entity
    item = {
        "id": "item1",
        "name": "Magic Sword",
        "description": "A powerful magical sword",
        "properties": {"damage": "10", "magic": "fire"},
        "usage": "Combat",
        "value": "1000 gold",
    }

    # Preprocess the entity
    text = entity_generator.preprocess_entity(item, "item")

    # Check if preprocessing produced expected text
    assert isinstance(text, str)
    assert "Magic Sword" in text
    assert "powerful magical sword" in text
    assert "Combat" in text
    assert "1000 gold" in text


def test_extract_entity_features(entity_generator):
    """Test extracting features from an entity."""
    # Create a test location entity
    location = {
        "id": "loc1",
        "name": "Ancient Forest",
        "description": "A mysterious forest with ancient trees",
        "environment": "Forest",
        "atmosphere": "Mysterious, ancient",
        "landmarks": ["Giant Oak", "Crystal Stream"],
        "connections": ["Mountain Pass", "Village"],
    }

    # Extract features
    features = entity_generator.extract_entity_features(location, "location")

    # Check if extraction produced expected features
    assert isinstance(features, dict)
    assert features["name"] == "Ancient Forest"
    assert features["description"] == "A mysterious forest with ancient trees"
    assert features["environment"] == "Forest"
    assert features["atmosphere"] == "Mysterious, ancient"
    assert isinstance(features["landmarks"], list)
    assert "Giant Oak" in features["landmarks"]
    assert isinstance(features["connections"], list)
    assert "Village" in features["connections"]


def test_entity_cache_key(entity_generator):
    """Test generation of entity cache key."""
    # Create test entities
    entity1 = {"id": "test1", "name": "Test Entity", "type": "character"}
    entity2 = {
        "id": "test1",
        "name": "Test Entity",
        "type": "character",
        "version": "v2",
    }

    # Generate cache keys
    key1 = entity_generator.get_entity_cache_key(entity1)
    key2 = entity_generator.get_entity_cache_key(entity2)

    # Keys should be strings and different
    assert isinstance(key1, str)
    assert isinstance(key2, str)
    assert key1 != key2  # Different because entity2 has version

    # Same entity should produce same key
    key1_duplicate = entity_generator.get_entity_cache_key(entity1)
    assert key1 == key1_duplicate
