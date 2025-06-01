"""
Integration tests for entity embedding generator functionality.

This module contains tests that verify the integration between
the EntityEmbeddingGenerator and the core embedding service.
"""

import pytest
import pytest_asyncio

from game_loop.embeddings.config import EmbeddingConfig
from game_loop.embeddings.entity_embeddings import EntityEmbeddingGenerator
from game_loop.embeddings.entity_registry import EntityEmbeddingRegistry
from game_loop.embeddings.service import EmbeddingService


class MockOllamaClient:
    """Mock OllamaClient for testing."""

    async def generate_embedding(self, text, model_name=None):
        """Mock embedding generation."""
        # Return deterministic embedding based on text length
        seed = len(text) % 10
        return [0.1 * (i % 10 + seed) / 10 for i in range(384)]

    async def generate_embeddings(self, text, config=None):
        """Mock generate_embeddings method (plural version)."""
        # This is the method called by EmbeddingService
        return await self.generate_embedding(text, model_name=None)


@pytest_asyncio.fixture
async def embedding_service():
    """Create an embedding service for testing."""
    config = EmbeddingConfig(
        model_name="nomic-embed-text",
        max_text_length=512,
        cache_enabled=True,
        cache_size=100,
    )

    # Create service with mock client
    mock_client = MockOllamaClient()
    service = EmbeddingService(embedding_config=config)
    service.client = mock_client
    return service


@pytest_asyncio.fixture
async def entity_generator(embedding_service):
    """Create an entity embedding generator for testing."""
    return EntityEmbeddingGenerator(embedding_service)


@pytest_asyncio.fixture
async def entity_registry():
    """Create an entity embedding registry for testing."""
    return EntityEmbeddingRegistry(dimension=384)


@pytest.mark.asyncio
async def test_end_to_end_entity_embedding(entity_generator, entity_registry):
    """Test end-to-end flow from entity to embedding to registry."""
    # Create test entities
    entities = [
        {
            "id": "char1",
            "name": "Adventurer",
            "description": "A brave explorer seeking fortune and glory.",
            "personality": "Bold and resourceful",
            "background": "Former soldier turned treasure hunter",
            "type": "character",
        },
        {
            "id": "loc1",
            "name": "Ancient Temple",
            "description": "A crumbling temple hidden in the jungle.",
            "environment": "Jungle, stone ruins",
            "atmosphere": "Mysterious, foreboding",
            "landmarks": ["Giant statue", "Hidden entrance", "Trap-filled corridor"],
            "type": "location",
        },
    ]

    # Generate embeddings for entities
    embeddings = []
    for entity in entities:
        embedding = await entity_generator.generate_entity_embedding(
            entity, entity["type"]
        )
        embeddings.append(embedding)

        # Register in the registry
        await entity_registry.register_entity(
            entity["id"], entity["type"], embedding, {"name": entity["name"]}
        )

    # Verify entities are in registry
    assert "char1" in entity_registry.get_all_entity_ids()
    assert "loc1" in entity_registry.get_all_entity_ids()

    # Test similarity search
    character_embedding = await entity_registry.get_entity_embedding("char1")
    similar = await entity_registry.find_similar_entities(character_embedding, top_k=2)

    # First result should be the character itself
    assert similar[0][0] == "char1"

    # Try batch processing
    batch_embeddings = await entity_generator.generate_entity_embeddings_batch(
        entities, entity_types=[e["type"] for e in entities]
    )

    # Verify batch results
    assert len(batch_embeddings) == 2
    assert all(len(emb) == 384 for emb in batch_embeddings)


@pytest.mark.asyncio
async def test_entity_type_specific_processing(entity_generator):
    """Test that different entity types are processed differently."""
    # Create entities of different types
    character = {
        "id": "char1",
        "name": "Wizard",
        "description": "A powerful magic user",
        "personality": "Wise and mysterious",
    }

    location = {
        "id": "loc1",
        "name": "Tower",
        "description": "A tall tower overlooking the valley",
        "environment": "Stone structure",
    }

    item = {
        "id": "item1",
        "name": "Staff",
        "description": "A magical wooden staff",
        "properties": {"damage": "5", "magic": "true"},
    }

    # Process each entity type
    char_text = entity_generator.preprocess_entity(character, "character")
    loc_text = entity_generator.preprocess_entity(location, "location")
    item_text = entity_generator.preprocess_entity(item, "item")

    # Verify each processing produces different results
    assert char_text != loc_text
    assert loc_text != item_text
    assert item_text != char_text

    # Check for type-specific content
    assert "Wizard" in char_text
    assert "personality" in char_text.lower()
    assert "Tower" in loc_text
    assert "environment" in loc_text.lower()
    assert "Staff" in item_text
    assert "properties" in item_text.lower()
