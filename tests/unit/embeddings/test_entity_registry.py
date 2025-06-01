"""
Tests for entity embedding registry functionality.

This module contains unit tests for the EntityEmbeddingRegistry
and related entity embedding storage and search.
"""

import numpy as np
import pytest

from game_loop.embeddings.entity_registry import EntityEmbeddingRegistry


@pytest.fixture
def registry():
    """Create an entity embedding registry for testing."""
    return EntityEmbeddingRegistry(dimension=384)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    # Create deterministic but different embeddings
    np.random.seed(42)  # For reproducibility

    embeddings = {
        "char1": np.random.random(384).tolist(),
        "char2": np.random.random(384).tolist(),
        "loc1": np.random.random(384).tolist(),
        "loc2": np.random.random(384).tolist(),
        "item1": np.random.random(384).tolist(),
    }

    # Make char1 and char2 more similar to each other
    base = np.random.random(384)
    embeddings["char1"] = (base + 0.1 * np.random.random(384)).tolist()
    embeddings["char2"] = (base + 0.1 * np.random.random(384)).tolist()

    return embeddings


@pytest.mark.asyncio
async def test_register_and_retrieve_entity(registry):
    """Test registering and retrieving an entity embedding."""
    entity_id = "test1"
    entity_type = "character"
    embedding = [0.1] * 384
    metadata = {"name": "Test Entity", "version": "1.0"}

    # Register the entity
    await registry.register_entity(entity_id, entity_type, embedding, metadata)

    # Retrieve the embedding
    retrieved = await registry.get_entity_embedding(entity_id)

    # Check if retrieved embedding matches registered embedding
    assert retrieved == embedding
    assert registry.entity_types[entity_id] == entity_type
    assert registry.metadata[entity_id] == metadata


@pytest.mark.asyncio
async def test_find_similar_entities(registry, sample_embeddings):
    """Test finding similar entities by embedding."""
    # Register sample entities
    await registry.register_entity("char1", "character", sample_embeddings["char1"])
    await registry.register_entity("char2", "character", sample_embeddings["char2"])
    await registry.register_entity("loc1", "location", sample_embeddings["loc1"])
    await registry.register_entity("loc2", "location", sample_embeddings["loc2"])
    await registry.register_entity("item1", "item", sample_embeddings["item1"])

    # Find similar entities to char1
    similar = await registry.find_similar_entities(sample_embeddings["char1"], top_k=3)

    # Check if results are in expected format
    assert isinstance(similar, list)
    assert len(similar) == 3
    assert similar[0][0] == "char1"  # Most similar to itself
    assert isinstance(similar[0][1], float)
    assert similar[0][1] > 0.9  # High similarity to self

    # Check filtering by entity type
    char_similar = await registry.find_similar_entities(
        sample_embeddings["char1"], entity_type="character", top_k=2
    )
    assert len(char_similar) == 2
    assert all(
        registry.entity_types[entity_id] == "character" for entity_id, _ in char_similar
    )


@pytest.mark.asyncio
async def test_update_entity_embedding(registry):
    """Test updating an entity embedding."""
    entity_id = "test1"
    initial_embedding = [0.1] * 384
    updated_embedding = [0.2] * 384

    # Register the entity
    await registry.register_entity(entity_id, "character", initial_embedding)

    # Update the embedding
    await registry.update_entity_embedding(entity_id, updated_embedding)

    # Retrieve the updated embedding
    retrieved = await registry.get_entity_embedding(entity_id)

    # Check if retrieved embedding matches updated embedding
    assert retrieved == updated_embedding


@pytest.mark.asyncio
async def test_remove_entity(registry):
    """Test removing an entity from the registry."""
    entity_id = "test1"
    embedding = [0.1] * 384

    # Register the entity
    await registry.register_entity(entity_id, "character", embedding)

    # Remove the entity
    result = await registry.remove_entity(entity_id)

    # Check if removal was successful
    assert result is True
    assert await registry.get_entity_embedding(entity_id) is None

    # Try removing non-existent entity
    result = await registry.remove_entity("non_existent")
    assert result is False


@pytest.mark.asyncio
async def test_get_all_entity_ids(registry):
    """Test retrieving all entity IDs from the registry."""
    # Register test entities
    await registry.register_entity("char1", "character", [0.1] * 384)
    await registry.register_entity("char2", "character", [0.1] * 384)
    await registry.register_entity("loc1", "location", [0.1] * 384)

    # Get all entity IDs
    all_ids = registry.get_all_entity_ids()

    # Check if all expected IDs are returned
    assert len(all_ids) == 3
    assert "char1" in all_ids
    assert "char2" in all_ids
    assert "loc1" in all_ids

    # Get filtered entity IDs
    character_ids = registry.get_all_entity_ids(entity_type="character")

    # Check if filtered IDs are correct
    assert len(character_ids) == 2
    assert "char1" in character_ids
    assert "char2" in character_ids
    assert "loc1" not in character_ids


@pytest.mark.asyncio
async def test_export_import_registry(registry, tmp_path):
    """Test exporting and importing the registry."""
    # Register test entities
    await registry.register_entity(
        "char1", "character", [0.1] * 384, {"name": "Character 1"}
    )
    await registry.register_entity(
        "loc1", "location", [0.2] * 384, {"name": "Location 1"}
    )

    # Export the registry
    export_path = tmp_path / "registry.pkl"
    await registry.export_registry(export_path)

    # Create a new registry and import the data
    new_registry = EntityEmbeddingRegistry(dimension=384)
    await new_registry.import_registry(export_path)

    # Check if imported registry has same data
    assert len(new_registry.embeddings) == 2
    assert "char1" in new_registry.embeddings
    assert "loc1" in new_registry.embeddings
    assert new_registry.entity_types["char1"] == "character"
    assert new_registry.entity_types["loc1"] == "location"
    assert new_registry.metadata["char1"] == {"name": "Character 1"}
